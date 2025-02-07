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
import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F

from car.distributions.distribution import ContinuosDistribution


class SquashedGaussian(ContinuosDistribution):
    _mean: torch.Tensor
    _var: torch.Tensor

    def __init__(self, mean: torch.Tensor, ln_sigma: torch.Tensor):
        super().__init__()
        assert mean.shape == ln_sigma.shape
        self._mean = mean
        self._std = torch.exp(ln_sigma)
        self._ln_sigma = ln_sigma
        self._batch_size = mean.shape[0]
        self._data_dim = mean.shape[1:]
        self._ndim = mean.shape[-1]
        self._dist = D.Normal(self._mean, self._std)

    @property
    def ndim(self):
        return self._ndim

    def sample(self):
        return torch.tanh(self._dist.rsample())

    def sample_and_log_prob(self):
        x = self._dist.rsample()
        return torch.tanh(x), self._log_prob_internal(x)

    def most_probable(self):
        return torch.tanh(self._mean)

    def mean(self):
        return torch.tanh(self._mean)

    def log_prob(self, x):
        x = torch.arctanh(x)
        return self._log_prob_internal(x)

    def _log_prob_internal(self, x):
        dim = len(x.shape) - 1
        gaussian_part = torch.sum(self._dist.log_prob(x), dim=dim, keepdim=True)
        adjust_part = torch.sum(self._log_determinant_jacobian(x), dim=dim, keepdim=True)
        return gaussian_part - adjust_part

    def _log_determinant_jacobian(self, x):
        # arctanh(y)' = 1/(1 - y^2) (y=tanh(x))
        # Below computes log(1 - tanh(x)^2)
        # For derivation see:
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py
        return 2.0 * (np.log(2.0) - x - F.softplus(-2.0 * x))
