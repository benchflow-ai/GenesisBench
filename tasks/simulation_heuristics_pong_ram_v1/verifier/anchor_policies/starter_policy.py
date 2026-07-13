# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weak late reactive Pong RAM controller for GenesisBench.

The RAM decoding constants are derived from
Trinkle23897/learning-beyond-gradients at commit
3555c2956c257d49a5015b782cbe485b14fd659e.
"""

from __future__ import annotations

import numpy as np


HOME_Y = 105.0
CHASE_AFTER_X = 122.0
DEADBAND = 10.0


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def reset(self, seed: int = 0) -> None:
        self.seed = seed

    def act(self, observation: np.ndarray) -> int:
        ram = np.asarray(observation, dtype=np.uint8)
        if ram.shape != (128,):
            raise ValueError(f"Expected Pong RAM shape (128,), got {ram.shape}")

        self_y = 0.972157 * float(ram[51]) - 2.553996
        target_y = HOME_Y
        if int(ram[54]) != 0:
            ball_x = float(ram[49]) - 49.0
            if ball_x > CHASE_AFTER_X:
                target_y = float(ram[54]) - 13.0

        error = target_y - self_y
        if error < -DEADBAND:
            return 2
        if error > DEADBAND:
            return 3
        return 0
