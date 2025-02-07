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
from typing import Tuple

import torch


class Distribution:
    def sample(self):
        """Sample a value from the distribution.

        Returns:
            torch.Tensor: Sampled value
        """
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        """The number of dimensions of the distribution."""
        raise NotImplementedError

    def choose_probable(self) -> torch.Tensor:
        """Compute the most probable action of the distribution.

        Returns:
            torch.Tensor: Probable action of the distribution
        """
        raise NotImplementedError

    def mean(self) -> torch.Tensor:
        """Compute the mean of the distribution (if exist)

        Returns:
            torch.Tensor: mean of the distribution

        Raises:
             NotImplementedError: The distribution does not have mean
        """
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of given input.

        Args:
            x (torch.Tensor): Target value to compute the log probability

        Returns:
            torch.Tensor: Log probability of given input
        """
        raise NotImplementedError

    def sample_and_log_prob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a value from the distribution and compute its log probability.

        Returns:
            Tuple[torch.Tensor]: Sampled value and its log probabilty
        """
        raise NotImplementedError

    def entropy(self) -> torch.Tensor:
        """Compute the entropy of the distribution.

        Returns:
            torch.Tensor: Entropy of the distribution
        """
        raise NotImplementedError


class DiscreteDistribution(Distribution):
    pass


class ContinuosDistribution(Distribution):
    pass
