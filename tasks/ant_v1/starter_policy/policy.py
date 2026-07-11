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

"""Weak rhythmic CPG/PD starter policy for GenesisBench Ant v1."""

from __future__ import annotations

import math

import numpy as np


Q_INDEX = np.asarray([6, 7, 0, 1, 2, 3, 4, 5], dtype=np.int64)
LEG_PHASE = np.asarray([0.0, math.pi, 0.0, math.pi], dtype=np.float64)
HIP_SIGN = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
ANKLE_SIGN = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.phase = 0.0

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.phase = 0.0

    def act(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        q = observation[5:13][Q_INDEX]
        dq = observation[19:27][Q_INDEX]

        leg_phase = self.phase + LEG_PHASE
        hip_target = HIP_SIGN * (
            0.1423 + 0.3020 * np.sin(leg_phase)
        )
        ankle_target = ANKLE_SIGN * (
            0.3845 + 0.3498 * np.cos(leg_phase)
        )

        action = np.empty(8, dtype=np.float64)
        action[0::2] = 0.4583 * (hip_target - q[0::2])
        action[1::2] = 0.4583 * (ankle_target - q[1::2])
        action -= 0.0 * dq

        self.phase += 0.4236
        return np.clip(action, -1.0, 1.0)

