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

"""Article-faithful 387-point RAM starter for Breakout."""

from __future__ import annotations

import numpy as np


NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3

FIELD_LEFT = 8.0
FIELD_RIGHT = 151.0
PADDLE_MIN_X = 15.5
PADDLE_MAX_X = 152.5
PADDLE_Y = 189.5
HOME_X = 106.5
DEADBAND = 3.0
CHASE_LEAD_STEPS = 6.0
LAUNCH_OFFSET = 24.0
MAX_VELOCITY_JUMP = 24.0


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _reflect(value: float, lower: float, upper: float) -> float:
    span = upper - lower
    shifted = (value - lower) % (2.0 * span)
    if shifted <= span:
        return lower + shifted
    return upper - (shifted - span)


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.prev_ball: tuple[float, float] | None = None
        self.velocity: tuple[float, float] | None = None
        self.launch_sign = 1.0
        self.had_visible_ball = False
        self.ball_missing = True

    def act(self, observation: np.ndarray) -> int:
        ram = np.asarray(observation, dtype=np.uint8)
        if ram.shape != (128,):
            raise ValueError(f"Expected RAM shape (128,), got {ram.shape}")

        paddle_x = 1.005232 * float(ram[72]) - 39.797062
        ball = None
        if int(ram[101]) != 0:
            ball = (
                0.999043 * float(ram[99]) - 48.370898,
                0.993263 * float(ram[101]) + 11.227841,
            )
        self._update_ball(ball)

        if ball is None:
            target_x = HOME_X + self.launch_sign * LAUNCH_OFFSET
        else:
            target_x = self._target_x(ball)

        error = target_x - paddle_x
        if error > DEADBAND:
            return RIGHT
        if error < -DEADBAND:
            return LEFT
        return FIRE if ball is None else NOOP

    def _update_ball(self, ball: tuple[float, float] | None) -> None:
        if ball is None:
            if self.had_visible_ball and not self.ball_missing:
                self.launch_sign *= -1.0
            self.ball_missing = True
            self.prev_ball = None
            self.velocity = None
            return

        self.had_visible_ball = True
        self.ball_missing = False
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

    def _target_x(self, ball: tuple[float, float]) -> float:
        ball_x, ball_y = ball
        if self.velocity is None:
            return _clip(ball_x, PADDLE_MIN_X, PADDLE_MAX_X)

        vx, vy = self.velocity
        if vy > 0.1 and ball_y <= PADDLE_Y:
            steps_to_paddle = max((PADDLE_Y - ball_y) / vy, 0.0)
            target_x = _reflect(
                ball_x + vx * steps_to_paddle,
                FIELD_LEFT,
                FIELD_RIGHT,
            )
        else:
            target_x = ball_x + CHASE_LEAD_STEPS * vx
        return _clip(target_x, PADDLE_MIN_X, PADDLE_MAX_X)
