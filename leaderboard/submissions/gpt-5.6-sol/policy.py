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

"""Robust rhythmic locomotion controller for Gymnasium Ant-v5."""

from __future__ import annotations

import math

import numpy as np


Q_INDEX = np.asarray([6, 7, 0, 1, 2, 3, 4, 5], dtype=np.int64)
LEG_PHASE = np.asarray([0.0, math.pi, 0.0, math.pi], dtype=np.float64)
HIP_SIGN = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
ANKLE_SIGN = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.phase = 0.0

    def act(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        q = observation[5:13][Q_INDEX]
        dq = observation[19:27][Q_INDEX]
        leg_phase = self.phase + LEG_PHASE

        target = np.empty(8, dtype=np.float64)
        target[0::2] = HIP_SIGN * (
            0.31809120 + 0.60658591 * np.sin(leg_phase)
        )
        target[1::2] = ANKLE_SIGN * (
            0.45889793 + 0.51872773 * np.sin(leg_phase + 0.59169138)
        )

        action = np.empty(8, dtype=np.float64)
        action[0::2] = (
            0.85014303 * (target[0::2] - q[0::2])
            - 0.06779820 * dq[0::2]
        )
        action[1::2] = (
            0.39878097 * (target[1::2] - q[1::2])
            - 0.13557792 * dq[1::2]
        )

        w, x, y, z = observation[1:5]
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        action[0::2] -= (
            0.30344660 * yaw
            - 0.00286854 * observation[18]
            - 0.03553088 * observation[14]
        )

        roll_feedback = 0.08478651 * roll + 0.02845704 * observation[16]
        pitch_feedback = -0.08075882 * pitch + 0.03412520 * observation[17]
        extension_feedback = (
            -0.04813340 * (0.55 - observation[0])
            + 0.03745279 * observation[15]
            + 0.00579829 * observation[13]
        )
        action[1] += -roll_feedback + pitch_feedback + extension_feedback
        action[3] += roll_feedback + pitch_feedback + extension_feedback
        action[5] += -roll_feedback + pitch_feedback - extension_feedback
        action[7] += roll_feedback + pitch_feedback - extension_feedback

        self.phase += 0.45097650
        return np.clip(action, -1.0, 1.0)

