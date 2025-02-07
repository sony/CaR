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
import argparse
import os
import pathlib
import random as py_random
from typing import Dict, Sequence

import gymnasium
import numpy as np
import torch

from car.algorithms.qrsac import QRSAC, QRSACConfig
from car.algorithms.qrsac_car import QRSACCaR, QRSACCaRConfig
from car.environments.wrapppers import (
    AppendPendulumEpisodeConstraintWrapper,
    AppendPendulumTimestepConstraintWrapper,
    RemoveRewardWrapper,
)
from car.file_writer import FileWriter


class _Solver:
    def __init__(self, alpha=1.0):
        self._alpha = alpha

    def compute_update(self, key: str, param: float, grad: float):
        raise NotImplementedError


class _AdamSolver(_Solver):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(alpha)
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps

        self._m = {}
        self._v = {}
        self._t = {}

    def compute_update(self, key: str, param: float, grad: float):
        if key not in self._m:
            self._m[key] = 0
        if key not in self._v:
            self._v[key] = 0
        if key not in self._t:
            self._t[key] = 1

        self._m[key] = self._beta1 * self._m[key] + (1.0 - self._beta1) * grad
        self._v[key] = self._beta2 * self._v[key] + (1.0 - self._beta2) * grad**2
        self._t[key] += 1

        bias_correction = np.sqrt(1.0 - self._beta2 ** self._t[key]) / (1.0 - self._beta1 ** self._t[key])
        alpha = self._alpha * bias_correction
        # NOTE: negative sign is necessary because we are solving a maximization problem instead of a minimization
        return param - alpha * self._m[key] / (np.sqrt(self._v[key]) + self._eps)


class MultiplierUpdater:
    def __init__(self, solver):
        self._solver = solver

    def update_multipliers(self, algorithm, constraints: Dict[str, float]):
        new_multipliers = {}
        old_multipliers = algorithm.current_multipliers()
        for key, grad in constraints.items():
            param = old_multipliers.get(key, 0.0)
            new_multiplier = self._solver.compute_update(key, param, grad)
            new_multipliers[key] = max(new_multiplier, 0.0)
        algorithm.update_multipliers(new_multipliers)


def build_qrsac(env, args):
    config = QRSACConfig(args.gpu, start_timesteps=args.start_timesteps)
    return QRSAC(env, config)


def build_qrsac_car(env, args):
    config = QRSACCaRConfig(args.gpu, start_timesteps=args.start_timesteps)
    return QRSACCaR(env, config)


def build_algorithm(env, args):
    if args.without_car:
        return build_qrsac(env, args)
    else:
        return build_qrsac_car(env, args)


def build_env(env_name, args, *, training=True):
    env = gymnasium.make(env_name)
    if args.without_car:
        return env

    if args.constraint_type == "episode":
        env = AppendPendulumEpisodeConstraintWrapper(env)
    elif args.constraint_type == "timestep":
        env = AppendPendulumTimestepConstraintWrapper(env)
    else:
        raise NotImplementedError(f"Unknown constraint type: {args.constraint_type}")
    if training:
        # NOTE: Ensure rewards are not fed to training algorithm
        env = RemoveRewardWrapper(env)
    return env


def run_one_episode(env, algorithm):
    rewards = []
    constraints = {}
    state, _ = env.reset()
    action = algorithm.compute_eval_action(state, begin_of_episode=True)
    while True:
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        for key, value in info.items():
            if "constraint" not in key:
                continue
            if key not in constraints:
                constraints[key] = []
            constraints[key].append(value)

        rewards.append(reward)
        if done:
            break
        else:
            state = next_state
            action = algorithm.compute_eval_action(state, begin_of_episode=False)
    return np.sum(rewards), constraints


def run_evaluation(env, algorithm):
    num_runs = 10

    def _average_constraints(episode_constraints: Sequence[Dict[str, float]]):
        def _compute_average(key, value):
            # NOTE: np.mean is a bit different from discounted expectation but we use mean for simplicity
            return np.sum(value) / num_runs if "episode" in key else np.mean(value)

        averages: Dict[str, Sequence] = {}
        for constraints in episode_constraints:
            for key, value in constraints.items():
                if key not in averages:
                    averages[key] = value
                else:
                    averages[key].extend(value)
        return {key: _compute_average(key, value) for key, value in averages.items()}

    print(f"evaluation @{algorithm._iteration_num}")
    scores = []
    constraints = []
    for run in range(num_runs):
        score, episode_constraints = run_one_episode(env, algorithm)
        scores.append(score)
        constraints.append(episode_constraints)
        print(f"run #{run + 1}: {score}")

    return scores, _average_constraints(constraints)


def prepare_output_dir(args, time_format="%Y-%m-%d-%H%M%S"):
    import datetime

    time_str = datetime.datetime.now().strftime(time_format)
    prefix = f"qrsac/{'without' if args.without_car else 'with'}-car"
    outdir = os.path.join(os.path.abspath(args.save_dir), f"training_results/{prefix}/{time_str}")
    outdir = pathlib.Path(outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)
    return outdir


def run_training(args):
    train_env = build_env(args.env, args, training=True)
    eval_env = build_env(args.env, args, training=False)

    algorithm = build_algorithm(train_env, args)

    multiplier_updater = MultiplierUpdater(_AdamSolver(alpha=args.multiplier_lr))
    outdir = prepare_output_dir(args)
    print(f"outdir: {outdir}")
    result_writer = FileWriter(outdir=outdir, file_prefix="evaluation_result")

    for i in range(args.total_iterations):
        algorithm.train(train_env)

        if (i + 1) % args.eval_timing == 0:
            scores, constraints = run_evaluation(eval_env, algorithm)
            if not args.without_car:
                multiplier_updater.update_multipliers(algorithm, constraints)
            result_writer.write_scalar(iteration_num=i + 1, scalar={"score": np.mean(scores)})


def main():
    parent_dir = str(pathlib.Path(__file__).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--total-iterations", type=int, default=100000)
    parser.add_argument("--eval-timing", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default=parent_dir)
    parser.add_argument("--start-timesteps", type=int, default=1000)

    # CaR specific settings
    parser.add_argument("--without-car", action="store_true")
    parser.add_argument("--constraint-type", type=str, default="episode", choices=["episode", "timestep"])
    parser.add_argument("--multiplier-lr", type=float, default=0.1)

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
