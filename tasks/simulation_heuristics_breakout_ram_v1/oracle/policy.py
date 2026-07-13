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

"""Frozen observation-only RAM reference for Breakout."""

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
FAST_LOW_BALL_LEAD_STEPS = 3.0
FAST_BALL_MIN_VY = 3.0
MAX_VELOCITY_JUMP = 24.0
STUCK_TRIGGER_STEPS = 1024
STUCK_SWITCH_STEPS = 256
STUCK_OFFSET = 12.0
STUCK_RELEASE_HORIZON = 8.0
BRICK_BALANCE_DEADZONE = 0.01
LATE_GAME_PADDLE_LAG = 2.0
LATE_GAME_LAG_BALL_Y = 170.0


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _reflect(value: float, lower: float, upper: float) -> float:
    span = upper - lower
    shifted = (value - lower) % (2.0 * span)
    if shifted <= span:
        return lower + shifted
    return upper - (shifted - span)


def _bit_count(values: np.ndarray) -> int:
    return int(sum(int(value).bit_count() for value in values.tolist()))


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.prev_ball: tuple[float, float] | None = None
        self.velocity: tuple[float, float] | None = None
        self.last_action = NOOP
        self.launch_sign = 1.0
        self.had_visible_ball = False
        self.ball_missing = True
        self.steps_since_progress = 0
        self.stuck_offset_index = 0
        self.previous_score: int | None = None
        self.score = 0

    def act(self, observation: np.ndarray) -> int:
        ram = np.asarray(observation, dtype=np.uint8)
        if ram.shape != (128,):
            raise ValueError(f"Expected RAM shape (128,), got {ram.shape}")

        brick_bytes = ram[:36]
        self.score = self._decode_score(ram)
        self._update_progress(self.score)

        paddle_x = 1.005232 * float(ram[72]) - 39.797062
        ball = None
        if int(ram[101]) != 0:
            ball = (
                0.999043 * float(ram[99]) - 48.370898,
                0.993263 * float(ram[101]) + 11.227841,
            )
        self._update_ball(ball)

        if ball is None:
            action = self._serve(paddle_x)
        else:
            target_x = self._target_x(ball, self._brick_balance(brick_bytes))
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

    def _update_progress(self, score: int) -> None:
        if self.previous_score is not None and score != self.previous_score:
            self.steps_since_progress = 0
            self.stuck_offset_index = 0
        elif self.previous_score is not None:
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

        self.previous_score = score

    def _update_ball(self, ball: tuple[float, float] | None) -> None:
        if ball is None:
            if self.had_visible_ball and not self.ball_missing:
                self.launch_sign *= -1.0
                self.steps_since_progress = 0
                self.stuck_offset_index = 0
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

    def _serve(self, paddle_x: float) -> int:
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
            self.score < 432
            or self.velocity is None
            or self.velocity[1] <= 0.1
            or ball[1] < LATE_GAME_LAG_BALL_Y
        ):
            return paddle_x
        if self.last_action == RIGHT:
            return _clip(
                paddle_x + LATE_GAME_PADDLE_LAG,
                PADDLE_MIN_X,
                PADDLE_MAX_X,
            )
        if self.last_action == LEFT:
            return _clip(
                paddle_x - LATE_GAME_PADDLE_LAG,
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
        if self.score >= 432:
            if balance > BRICK_BALANCE_DEADZONE:
                direction = 1.0
            elif balance < -BRICK_BALANCE_DEADZONE:
                direction = -1.0
            release = _clip(
                steps_to_paddle / STUCK_RELEASE_HORIZON,
                0.0,
                1.0,
            )
            magnitude *= release
        return direction * magnitude

    @staticmethod
    def _brick_balance(brick_bytes: np.ndarray) -> float:
        return (
            _bit_count(brick_bytes[:18]) / 132.0
            - _bit_count(brick_bytes[18:36]) / 108.0
        )

    @staticmethod
    def _decode_score(ram: np.ndarray) -> int:
        low_bcd = int(ram[77])
        return (
            100 * int(ram[76])
            + 10 * (low_bcd >> 4)
            + (low_bcd & 0x0F)
        )
