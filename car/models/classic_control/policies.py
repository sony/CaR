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
import torch.nn.functional as F
import torch
from torch import nn

from car.distributions.squashed_gaussian import SquashedGaussian
from car.models.policy import StochasticPolicy


class PendulumPolicy(StochasticPolicy):
    def __init__(self, state_dim, action_dim, clip_log_sigma=True, min_log_sigma=-20.0, max_log_sigma=2.0):
        super().__init__()
        self._action_dim = action_dim
        self._clip_log_sigma = clip_log_sigma
        self._min_log_sigma = min_log_sigma
        self._max_log_sigma = max_log_sigma
        self._linear1 = nn.Linear(in_features=state_dim, out_features=256)
        self._linear2 = nn.Linear(in_features=256, out_features=256)
        self._linear_mean = nn.Linear(in_features=256, out_features=action_dim)
        self._linear_sigma = nn.Linear(in_features=256, out_features=action_dim)

        # Initialize the parameters with glorot and set the bias term to zero
        # This is the same as the initialization method used in nnabla (https://github.com/sony/nnabla)
        for layer in [self._linear1, self._linear2, self._linear_mean, self._linear_sigma]:
            self._initialize_with_glorot(layer)

    def pi(self, state):
        h = F.relu(self._linear1(state))
        h = F.relu(self._linear2(h))
        mean = self._linear_mean(h)
        ln_sigma = self._linear_sigma(h)
        if self._clip_log_sigma:
            ln_sigma = ln_sigma.clamp(self._min_log_sigma, self._max_log_sigma)

        return SquashedGaussian(mean=mean, ln_sigma=ln_sigma)

    def _initialize_with_glorot(self, module: nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.zeros_(module.bias)
