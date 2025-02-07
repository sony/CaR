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
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union, cast

import gymnasium
import numpy as np

State = Union[np.ndarray, Tuple[np.ndarray]]
Action = np.ndarray
NextState = State


@dataclass
class EnvironmentExplorerConfig:
    warmup_random_steps: int = 0
    reward_scalar: float = 1.0
    timelimit_as_terminal: bool = True
    initial_step_num: int = 0


class EnvironmentExplorer(metaclass=ABCMeta):
    """Base class for environment exploration methods."""

    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _config: EnvironmentExplorerConfig
    _state: Union[State, None]
    _action: Union[Action, None]
    _next_state: Union[NextState, None]
    _steps: int

    def __init__(self, config: EnvironmentExplorerConfig = EnvironmentExplorerConfig()):
        self._config = config

        self._state = None
        self._action = None
        self._next_state = None
        self._begin_of_episode = True

        self._steps = self._config.initial_step_num

    @abstractmethod
    def action(self, steps: int, state: np.ndarray, *, begin_of_episode: bool = False) -> Tuple[np.ndarray, Dict]:
        """Compute the action for given state at given timestep.

        Args:
            steps(int): timesteps since the beginning of exploration
            state(np.ndarray): current state to compute the action
            begin_of_episode(bool): Informs the beginning of episode. Used for rnn state reset.

        Returns:
            np.ndarray: action for current state at given timestep
        """
        raise NotImplementedError

    def step(self, env: gymnasium.Env, n: int = 1, break_if_done: bool = False):
        """Step n timesteps in given env.

        Args:
            env(gymnasium.Env): Environment
            n(int): Number of timesteps to act in the environment

        Returns:
            List[Experience]: List of experience.
                Experience consists of (state, action, reward, terminal flag, next state and extra info).
        """
        assert 0 < n
        experiences = []
        if self._state is None:
            self._state, *_ = env.reset()

        for _ in range(n):
            experience, done = self._step_once(env, begin_of_episode=self._begin_of_episode)
            experiences.append(experience)

            self._begin_of_episode = done
            if done and break_if_done:
                break
        return experiences

    def rollout(self, env: gymnasium.Env):
        """Rollout the episode in current env.

        Args:
            env(gymnasium.Env): Environment

        Returns:
            List[Experience]: List of experience.
                Experience consists of (state, action, reward, terminal flag, next state and extra info).
        """
        self._state, *_ = env.reset()

        done = False

        experiences = []
        while not done:
            experience, done = self._step_once(env, begin_of_episode=self._begin_of_episode)
            experiences.append(experience)
            self._begin_of_episode = done
        return experiences

    def _step_once(self, env, *, begin_of_episode=False):
        self._steps += 1
        if self._steps < self._config.warmup_random_steps:
            self._action, action_info = self._warmup_action(env, begin_of_episode=begin_of_episode)
        else:
            self._action, action_info = self.action(
                self._steps, cast(np.ndarray, self._state), begin_of_episode=begin_of_episode
            )
        self._next_state, r, terminated, truncated, step_info = env.step(self._action)
        done = terminated or truncated
        timelimit = truncated
        if _is_end_of_episode(done, timelimit, self._config.timelimit_as_terminal):
            non_terminal = 0.0
        else:
            non_terminal = 1.0

        extra_info: Dict[str, Any] = {}
        extra_info.update(action_info)
        extra_info.update(step_info)
        experience = (
            cast(np.ndarray, self._state),
            cast(np.ndarray, self._action),
            r * self._config.reward_scalar,
            non_terminal,
            cast(np.ndarray, self._next_state),
            extra_info,
        )

        if done:
            self._state, *_ = env.reset()
        else:
            self._state = self._next_state
        return experience, done

    def _warmup_action(self, env, *, begin_of_episode=False):
        return _sample_action(env)


def _is_end_of_episode(done, timelimit, timelimit_as_terminal):
    if not done:
        return False
    else:
        return (not timelimit) or (timelimit and timelimit_as_terminal)


def _sample_action(env):
    action_info: Dict[str, Any] = {}
    action = env.action_space.sample()
    return action, action_info
