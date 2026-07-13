# SPDX-License-Identifier: GPL-3.0-or-later
"""Asymmetric two-rate CPG with joint-space PD feedback."""

from __future__ import annotations

import math

import numpy as np


GAIT = np.asarray(
    [
        2.0720429776207037,
        2.0890962589603244,
        0.7454172634828156,
        -0.06577692937619559,
        0.6454089230776165,
        -0.6746024815060896,
        -0.5255276137795621,
        -0.776911900431121,
        0.4784639889085752,
        -1.5344226994990133,
        0.8642039688512475,
        -0.6508749004237321,
        -0.8139980805865409,
        -0.43273458554686073,
        0.9311256504045836,
        -0.3884941211535119,
        0.5058403771772146,
        0.29329785944557646,
        -0.41630444646878295,
        -0.17793318904003722,
        0.25287195521722133,
        0.1864602367775759,
        -0.2174643453198326,
        0.32053306369231493,
        0.18857020059058224,
        0.2877837436012836,
        -0.46494131953713835,
        -0.23362679675599532,
        -0.179250145837136,
        -0.2538333727428432,
        -0.19506084586961667,
        -0.3065628153040692,
    ],
    dtype=np.float64,
)
DT = 0.05
PHASE_SUBSTEPS = 10
KP = 1.0
KD = 0.02


def _frequency(raw_value: np.ndarray) -> np.ndarray:
    sigmoid = 1.0 / (1.0 + np.exp(-raw_value))
    return 0.5 + 4.5 * sigmoid


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.phase = 0.0

    def act(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        stance_frequency = _frequency(GAIT[0:1])
        swing_frequency = _frequency(GAIT[1:2])
        phase = np.asarray([self.phase], dtype=np.float64)

        for _ in range(PHASE_SUBSTEPS):
            frequency = np.where(
                np.sin(phase) > 0.0,
                swing_frequency,
                stance_frequency,
            )
            phase = (phase + 2.0 * math.pi * frequency * (DT / PHASE_SUBSTEPS)) % (
                2.0 * math.pi
            )
        self.phase = float(phase[0])

        features = np.asarray(
            [
                1.0,
                math.sin(self.phase),
                math.cos(self.phase),
                math.sin(2.0 * self.phase),
                math.cos(2.0 * self.phase),
            ],
            dtype=np.float64,
        )
        target = features @ GAIT[2:].reshape(5, 6)
        action = KP * (target - observation[2:8]) - KD * observation[11:17]
        return np.clip(action, -1.0, 1.0)
