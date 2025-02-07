# MIT License
#
# Copyright 2025 Sony Group Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from dataclasses import dataclass
from typing import Optional

import gymnasium
import numpy as np
import torch
from torch.optim import Adam

import car.functions as CF
from car.algorithms.algorithm import Algorithm, AlgorithmConfig
from car.algorithms.utils import sync_model
from car.environment_explorers.gaussian_policy_explorer import (
    GaussianPolicyExplorer,
    GaussianPolicyExplorerConfig,
)
from car.models.classic_control.distributional_functions import (
    ContinuousQuantileFunction,
    PendulumQuantileFunction,
)
from car.models.classic_control.policies import PendulumPolicy, StochasticPolicy
from car.replay_buffers.replay_buffer import ReplayBuffer
from car.replay_buffers.utils import marshal_experiences


@dataclass
class QRSACConfig(AlgorithmConfig):
    gamma: float = 0.99
    learning_rate: float = 3.0 * 1e-4
    batch_size: int = 256
    tau: float = 0.005
    environment_steps: int = 1
    gradient_steps: int = 1
    target_entropy: Optional[float] = None
    initial_temperature: Optional[float] = None
    fix_temperature: bool = False
    start_timesteps: int = 10000
    replay_buffer_size: int = 1000000

    # Quantile function settings
    num_quantiles: int = 32
    kappa: float = 1.0


class QRSAC(Algorithm):
    _env: gymnasium.Env
    _config: QRSACConfig
    _policy: StochasticPolicy
    _critic: ContinuousQuantileFunction

    def __init__(self, env: gymnasium.Env, config: QRSACConfig):
        super().__init__(config)
        self._env = env
        if config.kappa == 0.0:
            print("kappa is set to 0.0. Quantile regression loss will be used in the training")
        else:
            print("kappa is non 0.0. Quantile huber loss will be used in the training")

        state_dim = np.prod(env.observation_space.shape)
        action_dim = np.prod(env.action_space.shape)
        self._policy = PendulumPolicy(state_dim=state_dim, action_dim=action_dim).to(device=self._device)
        self._policy_optimizer = Adam(self._policy.parameters(), lr=config.learning_rate)
        self._critics = [
            PendulumQuantileFunction(state_dim=state_dim, action_dim=action_dim, n_quantile=config.num_quantiles).to(
                device=self._device
            )
            for _ in range(2)
        ]
        self._critic_optimizers = [Adam(critic.parameters(), lr=config.learning_rate) for critic in self._critics]

        self._target_critics = [
            PendulumQuantileFunction(state_dim=state_dim, action_dim=action_dim, n_quantile=config.num_quantiles).to(
                device=self._device
            )
            for _ in range(2)
        ]
        for critic, target_critic in zip(self._critics, self._target_critics):
            sync_model(critic, target_critic)

        self._tau_hat = torch.FloatTensor(self._precompute_tau_hat(config.num_quantiles)).to(device=self._device)

        entropy = config.target_entropy if config.target_entropy else -np.prod(env.action_space.shape)
        self._target_entropy = torch.FloatTensor([entropy]).to(device=self._device)

        temperature = config.initial_temperature if config.initial_temperature else np.random.normal()
        self._log_temperature = torch.FloatTensor([temperature]).to(device=self._device)
        self._log_temperature.requires_grad = True
        if config.fix_temperature:
            self._temperature_optimizer = None
        else:
            self._temperature_optimizer = Adam([self._log_temperature], lr=config.learning_rate)

        self._explorer = GaussianPolicyExplorer(
            self._select_exploration_action,
            GaussianPolicyExplorerConfig(
                warmup_random_steps=config.start_timesteps,
                initial_step_num=self._iteration_num,
                timelimit_as_terminal=False,
            ),
        )
        self._replay_buffer = ReplayBuffer(capacity=config.replay_buffer_size)

    def compute_eval_action(self, state, *, begin_of_episode):
        action, _ = self._select_evaluation_action(state, begin_of_episode=begin_of_episode)
        return action

    def _run_online_training_iteration(self, env):
        for _ in range(self._config.environment_steps):
            self._run_environment_step(env)
        for _ in range(self._config.gradient_steps):
            self._run_gradient_step(self._replay_buffer)

    def _run_environment_step(self, env):
        experiences = self._explorer.step(env)
        self._replay_buffer.append_all(experiences)

    def _run_gradient_step(self, replay_buffer):
        if self._config.start_timesteps < self._iteration_num:
            self._qrsac_training(replay_buffer)

    def _qrsac_training(self, replay_buffer):
        experiences, *_ = replay_buffer.sample(self._config.batch_size, num_steps=1)
        (s, a, r, non_terminal, s_next, *_) = marshal_experiences(experiences)

        batch = (
            torch.FloatTensor(s).to(self._device),
            torch.FloatTensor(a).to(self._device),
            torch.FloatTensor(r).to(self._device),
            torch.FloatTensor(non_terminal).to(self._device),
            torch.FloatTensor(s_next).to(self._device),
        )

        self._quantile_function_training(batch)
        for critic, target_critic in zip(self._critics, self._target_critics):
            sync_model(critic, target_critic, tau=self._config.tau)

        self._policy_training(batch)

    def _quantile_function_training(self, batch):
        s_current, action, reward, non_terminal, s_next = batch

        with torch.no_grad():
            a_next, log_pi = self._policy(s_next).sample_and_log_prob()
            next_qs = []
            for target_critic in self._target_critics:
                q_function = target_critic.as_q_function()
                next_qs.append(q_function(s_next, a_next))
            next_qs = torch.concatenate(next_qs, dim=-1)
            min_indices = torch.argmin(next_qs, dim=-1, keepdim=True)
            min_indices = min_indices.unsqueeze(1).repeat(1, self._config.num_quantiles, 1)

            theta_js = []
            for target_critic in self._target_critics:
                theta_js.append(target_critic(s_next, a_next))
            theta_js = torch.stack(theta_js, dim=-1)
            theta_j = torch.gather(theta_js, dim=2, index=min_indices).squeeze()

            temperature = self._log_temperature.exp()
            Ttheta_j = reward + self._config.gamma * non_terminal * (theta_j - temperature * log_pi)
            Ttheta_j = Ttheta_j.unsqueeze(1)
            assert Ttheta_j.shape == (self._config.batch_size, 1, self._config.num_quantiles)

        with torch.no_grad():
            tau_hat = torch.unsqueeze(self._tau_hat, 0)
            tau_hat = tau_hat.repeat(self._config.batch_size, 1)
            tau_hat = torch.unsqueeze(tau_hat, 2)
            assert tau_hat.shape == (self._config.batch_size, self._config.num_quantiles, 1)

        loss = 0
        for critic in self._critics:
            Ttheta_i = critic(s_current, action)
            Ttheta_i = torch.unsqueeze(Ttheta_i, 2)
            assert Ttheta_i.shape == tau_hat.shape
            quantile_loss = CF.quantile_huber_loss(Ttheta_j, Ttheta_i, self._config.kappa, tau_hat)
            assert quantile_loss.shape == (
                self._config.batch_size,
                self._config.num_quantiles,
                self._config.num_quantiles,
            )
            quantile_loss = torch.mean(quantile_loss, dim=2)
            quantile_loss = torch.sum(quantile_loss, dim=1)
            loss += torch.mean(quantile_loss)

        for optimizer in self._critic_optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self._critic_optimizers:
            optimizer.step()

    def _policy_training(self, batch):
        s_current, *_ = batch
        action, log_pi = self._policy(s_current).sample_and_log_prob()
        q_values = []
        for critic in self._critics:
            q_values.append(critic(s_current, action))
        min_q = torch.minimum(*q_values)
        pi_loss = torch.mean(self._log_temperature.exp() * log_pi - min_q)

        self._policy_optimizer.zero_grad()
        pi_loss.backward()
        self._policy_optimizer.step()

        if not self._config.fix_temperature:
            log_pi_detached = log_pi.detach()
            temperature_loss = -torch.mean(self._log_temperature.exp() * (log_pi_detached + self._target_entropy))
            self._temperature_optimizer.zero_grad()
            temperature_loss.backward()
            self._temperature_optimizer.step()

    def _select_exploration_action(self, state, *, begin_of_episode):
        state = torch.FloatTensor(state).to(self._device).unsqueeze(0)
        action = self._policy(state).sample()
        return np.squeeze(action.detach().cpu().numpy(), axis=0), {}

    def _select_evaluation_action(self, state, *, begin_of_episode):
        state = torch.FloatTensor(state).to(self._device).unsqueeze(0)
        action = self._policy(state).most_probable()
        return np.squeeze(action.detach().cpu().numpy(), axis=0), {}

    @staticmethod
    def _precompute_tau_hat(num_quantiles):
        tau_hat = [
            (tau_prev + tau_i) / num_quantiles / 2.0
            for tau_prev, tau_i in zip(range(0, num_quantiles), range(1, num_quantiles + 1))
        ]
        return np.array(tau_hat, dtype=np.float32)
