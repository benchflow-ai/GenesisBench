# SPDX-License-Identifier: GPL-3.0-or-later
"""Weak native-image explorer for Montezuma's Revenge."""

from __future__ import annotations

import numpy as np


NOOP = 0
RIGHTFIRE = 11
LEFTFIRE = 12


def _as_hwc(observation: np.ndarray) -> np.ndarray:
    frame = np.asarray(observation, dtype=np.uint8)
    if frame.ndim != 3:
        raise ValueError(f"Expected an RGB image, got shape {frame.shape}")
    if frame.shape[0] in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] < 3:
        raise ValueError(f"Expected RGB channels, got shape {frame.shape}")
    return frame[..., :3]


class Policy:
    """A valid but intentionally weak pixel-reactive starting point."""

    def __init__(self, seed: int = 0) -> None:
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.step = 0

    def act(self, observation: np.ndarray) -> int:
        frame = _as_hwc(observation)
        red = frame[..., 0].astype(np.int16)
        green = frame[..., 1].astype(np.int16)
        blue = frame[..., 2].astype(np.int16)
        warm_pixels = (red > 120) & (red > green + 30) & (red > blue + 30)

        self.step += 1
        if self.step <= 20:
            return NOOP

        locations = np.argwhere(warm_pixels)
        if locations.size:
            horizontal_center = float(np.mean(locations[:, 1]))
            return RIGHTFIRE if horizontal_center < frame.shape[1] / 2 else LEFTFIRE

        return RIGHTFIRE if (self.step // 48) % 2 == 0 else LEFTFIRE
