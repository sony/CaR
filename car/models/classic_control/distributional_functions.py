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
import torch
import torch.nn.functional as F
from torch import nn

from car.models.distributional_function import ContinuousQuantileFunction


class PendulumQuantileFunction(ContinuousQuantileFunction):
    def __init__(self, state_dim, action_dim, n_quantile):
        super().__init__()
        self._linear1 = nn.Linear(in_features=state_dim + action_dim, out_features=256)
        self._linear2 = nn.Linear(in_features=256, out_features=256)
        self._linear3 = nn.Linear(in_features=256, out_features=n_quantile)

        # Initialize the parameters with glorot and set the bias term to zero
        # This is the same as the initialization method used in nnabla (https://github.com/sony/nnabla)
        for layer in [self._linear1, self._linear2, self._linear3]:
            self._initialize_with_glorot(layer)

    def quantiles(self, state, action):
        h = torch.cat((state, action), dim=1)
        h = F.relu(self._linear1(h))
        h = F.relu(self._linear2(h))
        return self._linear3(h)

    def _initialize_with_glorot(self, module: nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.zeros_(module.bias)
