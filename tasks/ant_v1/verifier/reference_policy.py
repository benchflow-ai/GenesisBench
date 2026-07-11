# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Frozen improved rhythmic reference used to calibrate Ant v1."""

from __future__ import annotations

import math

import numpy as np


Q_INDEX = np.asarray([6, 7, 0, 1, 2, 3, 4, 5], dtype=np.int64)
LEG_PHASE = np.asarray([0.0, math.pi, 0.0, math.pi], dtype=np.float64)
HIP_SIGN = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
ANKLE_SIGN = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
HEADING_AXIS = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
PITCH_AXIS = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
ROLL_AXIS = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)


def _rpy(quaternion: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quaternion / (np.linalg.norm(quaternion) + 1e-12)
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(
        np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    )
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return roll, pitch, yaw


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self.phase = 0.0

    def act(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        q = observation[5:13][Q_INDEX]
        _, _, yaw = _rpy(observation[1:5])
        yaw_rate = float(observation[18])
        leg_phase = self.phase + LEG_PHASE

        hip_wave = 0.12217781430672398 + (
            0.3618286792202296 * np.sin(leg_phase)
            + 0.10975404801587477
            * np.sin(2.0 * leg_phase + 2.0862256065597453)
            + 0.04827596673280693
            * np.sin(3.0 * leg_phase - 0.4994408326343364)
        )
        ankle_wave = 0.36651046903486795 + (
            0.3226052803819716 * np.cos(leg_phase)
            - 0.003434817287963554
            * np.cos(2.0 * leg_phase + 1.2927488104774438)
            - 0.06968988354403895
            * np.cos(3.0 * leg_phase + 1.5873441034476188)
        )

        action = np.empty(8, dtype=np.float64)
        action[0::2] = 0.5493824068786054 * (
            HIP_SIGN * hip_wave
            + HEADING_AXIS
            * (-0.12067720879887742 * yaw + 0.04418873596679619 * yaw_rate)
            - q[0::2]
        )
        action[1::2] = 0.5493824068786054 * (
            ANKLE_SIGN * ankle_wave - q[1::2]
        )

        self.phase += 0.43096661190230784
        return np.clip(action, -1.0, 1.0)
