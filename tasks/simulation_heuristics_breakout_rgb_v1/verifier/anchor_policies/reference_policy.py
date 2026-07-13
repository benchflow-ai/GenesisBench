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

"""Frozen pixel-only reference for the Breakout RGB transfer experiment."""

from __future__ import annotations

import numpy as np


NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3

RED = np.asarray([200, 72, 72], dtype=np.uint8)
BLACK = np.asarray([0, 0, 0], dtype=np.uint8)
WALL = np.asarray([142, 142, 142], dtype=np.uint8)

FIELD_LEFT = 8.0
FIELD_RIGHT = 151.0
PADDLE_MIN_X = 15.5
PADDLE_MAX_X = 152.5
PADDLE_Y = 189.5
HOME_X = 106.5
DEADBAND = 3.0
CHASE_LEAD_STEPS = 8.0
LAUNCH_OFFSET = 24.0
FAST_LOW_BALL_LEAD_STEPS = 3.0
FAST_BALL_MIN_VY = 3.0
MAX_VELOCITY_JUMP = 24.0
STUCK_TRIGGER_STEPS = 1024
STUCK_SWITCH_STEPS = 256
STUCK_OFFSET = 12.0
STUCK_RELEASE_HORIZON = 8.0
BRICK_BALANCE_DEADZONE = 0.01
PADDLE_LAG = 2.0
LAG_BALL_Y = 170.0
MAX_MISSING_BALL_FRAMES = 8


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _reflect(value: float, lower: float, upper: float) -> float:
    span = upper - lower
    shifted = (value - lower) % (2.0 * span)
    if shifted <= span:
        return lower + shifted
    return upper - (shifted - span)


def _paddle_center(mask: np.ndarray, x_offset: int) -> float | None:
    best_length = 0
    best_center = None
    for row in mask:
        padded = np.pad(row.astype(np.int8), (1, 1))
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for start, end in zip(starts, ends, strict=True):
            length = int(end - start)
            if length > best_length:
                best_length = length
                best_center = x_offset + 0.5 * (start + end - 1)
    return float(best_center) if best_length >= 8 else None


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.prev_ball: tuple[float, float] | None = None
        self.velocity: tuple[float, float] | None = None
        self.last_action = NOOP
        self.launch_sign = 1.0
        self.steps_since_progress = 0
        self.stuck_offset_index = 0
        self.missing_ball_frames = 0
        self.previous_bricks: np.ndarray | None = None

    def act(self, observation: np.ndarray) -> int:
        pixels = np.asarray(observation, dtype=np.uint8)
        if pixels.shape != (3, 210, 160):
            raise ValueError(
                f"Expected RGB shape (3, 210, 160), got {pixels.shape}"
            )
        frame = np.moveaxis(pixels, 0, -1)

        ball, paddle_x, brick_balance, bricks = self._detect(frame)
        self._update_progress(bricks)
        self._update_ball(ball)

        if ball is None:
            action = self._serve(paddle_x)
        elif paddle_x is None:
            action = NOOP
        else:
            target_x = self._target_x(ball, brick_balance)
            control_x = self._control_paddle_x(paddle_x, ball)
            error = target_x - control_x
            if error > DEADBAND:
                action = RIGHT
            elif error < -DEADBAND:
                action = LEFT
            else:
                action = NOOP

        self.last_action = action
        return action

    def _detect(
        self,
        frame: np.ndarray,
    ) -> tuple[
        tuple[float, float] | None,
        float | None,
        float,
        np.ndarray,
    ]:
        red = np.all(frame == RED, axis=-1)
        paddle_x = _paddle_center(red[184:196, 8:152], 8)

        active = np.logical_and(
            ~np.all(frame == BLACK, axis=-1),
            ~np.all(frame == WALL, axis=-1),
        )
        active[:30, :] = False
        active[196:, :] = False
        active[:, :8] = False
        active[:, 152:] = False
        if paddle_x is not None:
            x0 = max(8, int(round(paddle_x)) - 12)
            x1 = min(152, int(round(paddle_x)) + 13)
            active[184:196, x0:x1] = False

        brick_active = active[57:93, 8:152]
        cells = brick_active.reshape(6, 6, 18, 8).sum(axis=(1, 3)) > 24
        occupied_pixels = np.repeat(
            np.repeat(cells, 6, axis=0),
            8,
            axis=1,
        )
        ball_mask = active.copy()
        ball_mask[57:93, 8:152] &= ~occupied_pixels
        ys, xs = np.nonzero(ball_mask)
        ball = None
        if (
            2 <= xs.size <= 8
            and int(xs.max() - xs.min() + 1) <= 4
            and int(ys.max() - ys.min() + 1) <= 6
        ):
            neighbors = np.zeros_like(ball_mask)
            neighbors[1:, :] |= ball_mask[:-1, :]
            neighbors[:-1, :] |= ball_mask[1:, :]
            neighbors[:, 1:] |= ball_mask[:, :-1]
            neighbors[:, :-1] |= ball_mask[:, 1:]
            touches_other_object = np.any(
                np.logical_and(
                    neighbors,
                    np.logical_and(active, ~ball_mask),
                )
            )
            if not touches_other_object:
                ball = (float(xs.mean()), float(ys.mean()))

        split = brick_active.shape[1] // 2
        left = float(np.count_nonzero(brick_active[:, :split]))
        right = float(np.count_nonzero(brick_active[:, split:]))
        brick_balance = (left - right) / max(left + right, 1.0)
        return ball, paddle_x, brick_balance, cells

    def _update_progress(self, bricks: np.ndarray) -> None:
        if self.previous_bricks is None:
            self.previous_bricks = bricks.copy()
            return
        if not np.array_equal(bricks, self.previous_bricks):
            self.steps_since_progress = 0
            self.stuck_offset_index = 0
        else:
            self.steps_since_progress += 1
            if (
                self.steps_since_progress >= STUCK_TRIGGER_STEPS
                and (
                    self.steps_since_progress - STUCK_TRIGGER_STEPS
                )
                % STUCK_SWITCH_STEPS
                == 0
            ):
                self.stuck_offset_index += 1
        self.previous_bricks = bricks.copy()

    def _update_ball(self, ball: tuple[float, float] | None) -> None:
        if ball is None:
            self.missing_ball_frames += 1
            if self.missing_ball_frames > MAX_MISSING_BALL_FRAMES:
                self.prev_ball = None
                self.velocity = None
            return

        self.missing_ball_frames = 0
        if self.prev_ball is not None:
            dx = ball[0] - self.prev_ball[0]
            dy = ball[1] - self.prev_ball[1]
            if (
                abs(dx) <= MAX_VELOCITY_JUMP
                and abs(dy) <= MAX_VELOCITY_JUMP
                and abs(dx) + abs(dy) > 0.25
            ):
                if self.velocity is None:
                    self.velocity = (dx, dy)
                else:
                    self.velocity = (
                        0.5 * self.velocity[0] + 0.5 * dx,
                        0.5 * self.velocity[1] + 0.5 * dy,
                    )
            else:
                self.velocity = None
        self.prev_ball = ball

    def _serve(self, paddle_x: float | None) -> int:
        if paddle_x is None:
            return FIRE
        target_x = HOME_X + self.launch_sign * LAUNCH_OFFSET
        error = target_x - paddle_x
        if error > DEADBAND:
            return RIGHT
        if error < -DEADBAND:
            return LEFT
        return FIRE

    def _target_x(self, ball: tuple[float, float], balance: float) -> float:
        ball_x, ball_y = ball
        if self.velocity is None:
            return _clip(ball_x, PADDLE_MIN_X, PADDLE_MAX_X)

        vx, vy = self.velocity
        if vy > 0.1 and ball_y <= PADDLE_Y:
            steps_to_paddle = max((PADDLE_Y - ball_y) / vy, 0.0)
            intercept = _reflect(
                ball_x + vx * steps_to_paddle,
                FIELD_LEFT,
                FIELD_RIGHT,
            )
            target_x = intercept + self._stuck_offset(
                steps_to_paddle,
                balance,
            )
        elif vy >= FAST_BALL_MIN_VY:
            target_x = ball_x + FAST_LOW_BALL_LEAD_STEPS * vx
        else:
            target_x = ball_x + CHASE_LEAD_STEPS * vx
        return _clip(target_x, PADDLE_MIN_X, PADDLE_MAX_X)

    def _control_paddle_x(
        self,
        paddle_x: float,
        ball: tuple[float, float],
    ) -> float:
        if (
            self.velocity is None
            or self.velocity[1] <= 0.1
            or ball[1] < LAG_BALL_Y
        ):
            return paddle_x
        if self.last_action == RIGHT:
            return _clip(
                paddle_x + PADDLE_LAG,
                PADDLE_MIN_X,
                PADDLE_MAX_X,
            )
        if self.last_action == LEFT:
            return _clip(
                paddle_x - PADDLE_LAG,
                PADDLE_MIN_X,
                PADDLE_MAX_X,
            )
        return paddle_x

    def _stuck_offset(self, steps_to_paddle: float, balance: float) -> float:
        if self.steps_since_progress < STUCK_TRIGGER_STEPS:
            return 0.0

        phase = self.stuck_offset_index % 4
        direction = 1.0 if phase in (0, 2) else -1.0
        magnitude = STUCK_OFFSET if phase in (0, 1) else 0.5 * STUCK_OFFSET
        if balance > BRICK_BALANCE_DEADZONE:
            direction = 1.0
        elif balance < -BRICK_BALANCE_DEADZONE:
            direction = -1.0
        release = _clip(
            steps_to_paddle / STUCK_RELEASE_HORIZON,
            0.0,
            1.0,
        )
        return direction * magnitude * release
