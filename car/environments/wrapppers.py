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

import gymnasium
import numpy as np


class RemoveRewardWrapper(gymnasium.RewardWrapper):
    def reward(self, _):
        return 0


class AppendPendulumConstraintWrapper(gymnasium.Wrapper, metaclass=ABCMeta):
    def step(self, u):
        th, thdot = self.unwrapped.state
        th = angle_normalize(th)
        next_state, reward, terminated, truncated, info = super().step(u)
        done = terminated or truncated
        info.update(self._compute_constraints(th, thdot, float(u), done))
        return next_state, reward, terminated, truncated, info

    @abstractmethod
    def _compute_constraints(self, th, thdot, u, done):
        raise NotImplementedError


class AppendPendulumTimestepConstraintWrapper(AppendPendulumConstraintWrapper):
    def _compute_constraints(self, th, thdot, u, done):
        # Eq.(17) in https://arxiv.org/abs/2501.04228
        # NOTE: Append prefix 'timestep' to the constraint's name for episodic constraints
        # to make the algorithm compute correct gradient
        return {"timestep-constraint-th": 0.01 - abs(th)}


class AppendPendulumEpisodeConstraintWrapper(AppendPendulumConstraintWrapper):
    def _compute_constraints(self, th, thdot, u, done):
        # Eq.(18) in https://arxiv.org/abs/2501.04228
        # NOTE: Append prefix 'episode' to the constraint's name for episodic constraints
        # to make the algorithm compute correct gradient
        return {"episode-constraint-th": 0.01 - abs(th) if done else 0.0}


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
