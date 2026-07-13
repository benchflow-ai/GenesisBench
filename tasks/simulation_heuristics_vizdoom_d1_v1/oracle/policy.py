"""Compact screen-CV reference for the VizDoom D1 article experiment.

This is an independent GenesisBench implementation of the published behavior.
No upstream source file is vendored.
"""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


NONE = 0
TURN_RIGHT = 1
TURN_LEFT = 2
FORWARD = 3
FORWARD_RIGHT = 4
FORWARD_LEFT = 5


def _medikit_box(
    frame: np.ndarray,
) -> tuple[float, float, float, float, float] | None:
    channel = np.asarray(frame)[..., 0]
    height, width = channel.shape
    mask = np.where(channel > 150, 255, 0).astype(np.uint8)
    mask[: int(height * 0.38), :] = 0
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 5), dtype=np.uint8),
    )
    mask = cv2.dilate(
        mask,
        np.ones((2, 3), dtype=np.uint8),
        iterations=1,
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

    best: tuple[float, float, float, float, float, float] | None = None
    for component in range(1, count):
        x, y, box_width, box_height, area = stats[component]
        center_x, center_y = centroids[component]
        aspect = box_width / max(1, box_height)
        if area < 10 or area > 3500:
            continue
        if y < height * 0.38 or box_height > height * 0.50:
            continue
        if box_width > width * 0.75 or not 0.35 <= aspect <= 10.0:
            continue
        if box_width > width * 0.40 and box_height < height * 0.05:
            continue
        score = float(area) + 0.9 * float(center_y)
        if 0.5 < aspect < 6.0:
            score += 30.0
        candidate = (
            score,
            float(area),
            float(center_x),
            float(center_y),
            float(box_width),
            float(box_height),
        )
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return None
    _, area, center_x, center_y, box_width, box_height = best
    return area, center_x, center_y, box_width, box_height


class Policy:
    _next_lane = 0

    def __init__(self) -> None:
        self._lane_phase = type(self)._next_lane
        type(self)._next_lane += 1
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.last_seen = -999
        self.last_direction = -1

    def act(
        self,
        frame: np.ndarray,
        variables: Mapping[str, float],
    ) -> int:
        image = np.asarray(frame)
        height, width = image.shape[:2]
        health = float(variables["HEALTH"])
        detected = _medikit_box(image)
        step = self.step
        self.step += 1

        if detected is None:
            if step - self.last_seen < 40:
                return TURN_LEFT if self.last_direction < 0 else TURN_RIGHT
            phase = (step // 35 + self._lane_phase) % 2
            return TURN_LEFT if phase == 0 else TURN_RIGHT

        area, center_x, center_y, box_width, box_height = detected
        offset = center_x - width / 2.0
        self.last_seen = step
        self.last_direction = -1 if center_x < width / 2.0 else 1
        close = (
            area > 180.0
            or center_y > height * 0.80
            or box_height > height * 0.12
            or box_width > width * 0.20
        )

        if close and health > 68.0:
            if offset < -16.0:
                return TURN_LEFT
            if offset > 16.0:
                return TURN_RIGHT
            return NONE

        margin = 18.0 if close else 12.0
        if offset < -margin:
            return FORWARD_LEFT
        if offset > margin:
            return FORWARD_RIGHT
        return FORWARD
