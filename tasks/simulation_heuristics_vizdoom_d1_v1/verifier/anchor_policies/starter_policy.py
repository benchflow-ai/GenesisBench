"""Frozen weak rendered-pixel anchor for VizDoom D1 Basic."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


NONE = 0
TURN_RIGHT = 1
TURN_LEFT = 2
FORWARD = 3
FORWARD_RIGHT = 4
FORWARD_LEFT = 5


class Policy:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.step = 0

    def act(
        self,
        frame: np.ndarray,
        variables: Mapping[str, float],
    ) -> int:
        self.step += 1
        health = float(variables["HEALTH"])
        if health > 25.0:
            return NONE

        image = np.asarray(frame)
        height, width = image.shape[:2]
        bright = image[..., 0] > 180
        bright[: height // 2] = False
        _, columns = np.nonzero(bright)

        if len(columns) < 8:
            return TURN_LEFT if (self.step // 45) % 2 == 0 else TURN_RIGHT

        center_x = float(columns.mean())
        offset = center_x - width / 2.0
        if offset < -14.0:
            return FORWARD_LEFT
        if offset > 14.0:
            return FORWARD_RIGHT
        return FORWARD
