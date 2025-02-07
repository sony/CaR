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
from collections import deque
from typing import Any, MutableSequence, Optional, Sequence, Tuple, Union, cast

import numpy as np

from car.replay_buffers.utils import DataHolder, RingBuffer

drng = np.random.default_rng()


class ReplayBuffer(object):
    # type declarations to type check with mypy
    # NOTE: declared variables are instance variable and NOT class variable, unless it is marked with ClassVar
    # See https://mypy.readthedocs.io/en/stable/class_basics.html for details
    _buffer: Union[MutableSequence[Any], DataHolder]

    def __init__(self, capacity: Optional[int] = None):
        assert capacity is None or capacity >= 0
        self._capacity = capacity

        if capacity is None:
            self._buffer = deque(maxlen=capacity)
        else:
            self._buffer = RingBuffer(maxlen=capacity)

    def __getitem__(self, item: int):
        return self._buffer[item]

    @property
    def capacity(self) -> Union[int, None]:
        """Capacity (max length) of this replay buffer otherwise None."""
        return self._capacity

    def append(self, experience):
        """Add new experience to the replay buffer.

        Args:
            experience (array-like): Experience includes trainsitions,
                such as state, action, reward, the iteration of environment has done or not.
                Please see to get more information in [Replay buffer documents](replay_buffer.md)

        Notes:
            If the replay buffer size is full, the oldest (head of the buffer) experience will be dropped off
            and the given experince will be added to the tail of the buffer.
        """
        self._buffer.append(experience)

    def append_all(self, experiences):
        """Add list of experiences to the replay buffer.

        Args:
            experiences (Sequence[Experience]): Sequence of experiences to insert to the buffer

        Notes:
            If the replay buffer size is full, the oldest (head of the buffer) experience will be dropped off
            and the given experince will be added to the tail of the buffer.
        """
        for experience in experiences:
            self.append(experience)

    def sample(self, num_samples: int = 1, num_steps: int = 1):
        """Randomly sample num_samples experiences from the replay buffer.

        Args:
            num_samples (int): Number of samples to sample from the replay buffer. Defaults to 1.
            num_steps (int): Number of timesteps to sample. Should be greater than 0. Defaults to 1.

        Returns:
            experiences (Sequence[Experience] or Tuple[Sequence[Experience], ...]):
                Random num_samples of experiences. If num_steps is greater than 1, will return a tuple of size num_steps
                which contains num_samples of experiences for each entry.
            info (Dict[str, Any]): dictionary of information about experiences.

        Raises:
            ValueError: num_samples exceeds the maximum possible index or num_steps is 0 or negative.

        Notes
        ----
        Sampling strategy depends on undelying implementation.
        """
        max_index = len(self) - num_steps + 1
        if num_samples > max_index:
            raise ValueError(f"num_samples: {num_samples} is greater than the size of buffer: {max_index}")
        indices = self._random_indices(num_samples=num_samples, max_index=max_index)
        return self.sample_indices(indices, num_steps=num_steps)

    def sample_indices(self, indices: Sequence[int], num_steps: int = 1):
        """Sample experiences for given indices from the replay buffer.

        Args:
            indices (array-like): list of array index to sample the data
            num_steps (int): Number of timesteps to sample. Should not be negative. Defaults to 1.

        Returns:
            experiences (Sequence[Experience] or Tuple[Sequence[Experience], ...]):
                Random num_samples of experiences. If num_steps is greater than 1, will return a tuple of size num_steps
                which contains num_samples of experiences for each entry.
            info (Dict[str, Any]): dictionary of information about experiences.

        Raises:
            ValueError: If indices are empty or num_steps is 0 or negative.
        """
        if len(indices) == 0:
            raise ValueError("Indices are empty")
        if num_steps < 1:
            raise ValueError(f"num_steps: {num_steps} should be greater than 0!")
        experiences: Union[Sequence[Experience], Tuple[Sequence[Experience], ...]]
        if num_steps == 1:
            experiences = [self.__getitem__(index) for index in indices]
        else:
            experiences = tuple([self.__getitem__(index + i) for index in indices] for i in range(num_steps))
        weights = np.ones([len(indices), 1])
        return experiences, dict(weights=weights)

    def update_priorities(self, errors: np.ndarray):
        pass

    def __len__(self):
        return len(self._buffer)

    def _random_indices(self, num_samples: int, max_index: Optional[int] = None) -> Sequence[int]:
        if max_index is None:
            max_index = len(self)
        # NOTE: Do NOT replace with np.random.choice(max_index, size=num_samples, replace=False)
        # np.random.choice is terribly slow when sampling without replacement
        indices = drng.choice(max_index, size=num_samples, replace=False)
        return cast(Sequence[int], indices)
