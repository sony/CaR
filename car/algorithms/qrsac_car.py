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
import dataclasses
from dataclasses import dataclass, field
from typing import Dict

import gymnasium
import torch

from car.algorithms.qrsac import QRSAC, QRSACConfig
from car.algorithms.utils import sync_model
from car.replay_buffers.utils import marshal_experiences


@dataclass
class QRSACCaRConfig(QRSACConfig):
    multipliers: Dict[str, float] = field(default_factory=dict)


class QRSACCaR(QRSAC):
    _env: gymnasium.Env
    _config: QRSACCaRConfig

    def __init__(self, env: gymnasium.Env, config: QRSACCaRConfig):
        super().__init__(env=env, config=config)

    def current_multipliers(self):
        return self._config.multipliers

    def update_multipliers(self, new_multipliers: Dict[str, float]):
        self._config = dataclasses.replace(self._config, multipliers=new_multipliers)

    def _qrsac_training(self, replay_buffer):
        experiences, *_ = replay_buffer.sample(self._config.batch_size, num_steps=1)
        (s, a, r, non_terminal, s_next, info, *_) = marshal_experiences(experiences)

        batch = (
            torch.FloatTensor(s).to(self._device),
            torch.FloatTensor(a).to(self._device),
            torch.FloatTensor(self._build_reward(r, info)).to(self._device),
            torch.FloatTensor(non_terminal).to(self._device),
            torch.FloatTensor(s_next).to(self._device),
        )

        self._quantile_function_training(batch)
        for critic, target_critic in zip(self._critics, self._target_critics):
            sync_model(critic, target_critic, tau=self._config.tau)

        self._policy_training(batch)

    def _build_reward(self, reward, constraints):
        for key, multiplier in self._config.multipliers.items():
            if key in constraints:
                reward += constraints[key] * multiplier
        return reward
