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
from dataclasses import dataclass

from car.environment_explorers.environment_explorer import EnvironmentExplorer, EnvironmentExplorerConfig


@dataclass
class GaussianPolicyExplorerConfig(EnvironmentExplorerConfig):
    pass


class GaussianPolicyExplorer(EnvironmentExplorer):
    _config: GaussianPolicyExplorerConfig

    def __init__(self, action_selector, config=GaussianPolicyExplorerConfig()):
        super().__init__(config)
        self._action_selector = action_selector

    def action(self, steps, state, *, begin_of_episode=False):
        return self._action_selector(state, begin_of_episode=begin_of_episode)
