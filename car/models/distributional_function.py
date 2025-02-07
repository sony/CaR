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
from torch import nn

from car.models.q_function import QFunction


class QuantileDistributionFunction(nn.Module):
    def forward(self, state, action):
        return self.quantiles(state, action)

    def quantiles(self, state, action):
        raise NotImplementedError

    def as_q_function(self) -> QFunction:
        raise NotImplementedError


class ContinuousQuantileFunction(QuantileDistributionFunction):
    def as_q_function(self):
        class Wrapper(QFunction):
            _quantile_distribution_function: "ContinuousQuantileFunction"

            def __init__(self, quantile_distribution_function, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._quantile_distribution_function = quantile_distribution_function

            def q(self, state, action):
                quantiles = self._quantile_distribution_function.quantiles(state, action)
                return torch.mean(quantiles, dim=len(quantiles.shape) - 1, keepdim=True)

        return Wrapper(self)
