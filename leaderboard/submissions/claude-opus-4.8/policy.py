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

"""Tuned rhythmic CPG/PD controller for GenesisBench Ant v1.

A trot-gait central pattern generator drives per-joint sinusoidal targets that
are tracked by a stiff PD law with velocity damping. The continuous gait
parameters (offsets, amplitudes, gain, frequency, damping, hip/ankle phase)
were optimized with an evolution strategy against the native Ant reward across
a spread of conservative dynamics variants (weaker actuators, higher/lower
friction, heavier bodies, stronger damping) so the closed-loop controller
generalizes to unseen seeds and dynamics rather than memorizing episodes.
"""

from __future__ import annotations

import math

import numpy as np

# Reorder the 8 hinge joints so hips occupy the even lanes and ankles the odd
# lanes, matching the CPG target layout below.
Q_INDEX = np.asarray([6, 7, 0, 1, 2, 3, 4, 5], dtype=np.int64)
LEG_PHASE = np.asarray([0.0, math.pi, 0.0, math.pi], dtype=np.float64)
HIP_SIGN = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
ANKLE_SIGN = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)

# Optimized gait parameters:
# hip_off, hip_amp, ankle_off, ankle_amp, Kp, dphase, Kd, hip_ankle_phase
PARAMS = np.asarray(
    [0.07697, 0.32893, 0.44401, 0.23742, 1.03025, 0.43138, 0.02531, -0.18832],
    dtype=np.float64,
)


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.phase = 0.0
        (
            self.hip_off,
            self.hip_amp,
            self.ankle_off,
            self.ankle_amp,
            self.Kp,
            self.dphase,
            self.Kd,
            self.hap,
        ) = PARAMS

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.phase = 0.0

    def act(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        q = observation[5:13][Q_INDEX]
        dq = observation[19:27][Q_INDEX]

        leg_phase = self.phase + LEG_PHASE
        hip_target = HIP_SIGN * (self.hip_off + self.hip_amp * np.sin(leg_phase))
        ankle_target = ANKLE_SIGN * (
            self.ankle_off + self.ankle_amp * np.cos(leg_phase + self.hap)
        )

        action = np.empty(8, dtype=np.float64)
        action[0::2] = self.Kp * (hip_target - q[0::2])
        action[1::2] = self.Kp * (ankle_target - q[1::2])
        action -= self.Kd * dq

        self.phase += self.dphase
        return np.clip(action, -1.0, 1.0)
