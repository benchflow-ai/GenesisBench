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

"""Geometric Pong RAM controller reimplemented for GenesisBench.

The RAM decoding constants and controller behavior are derived from
Trinkle23897/learning-beyond-gradients at commit
3555c2956c257d49a5015b782cbe485b14fd659e. This file is a clean policy-only
implementation of that Apache-2.0-licensed artifact.
"""

from __future__ import annotations

import numpy as np


NOOP = 0
MOVE_UP = 2
MOVE_DOWN = 3
HOME_Y = 105.0
PADDLE_X = 141.0
FIELD_TOP = 34.0
FIELD_BOTTOM = 194.0
DEADBAND = 6.0
SPIN_OFFSET = 8.0
MAX_VELOCITY_JUMP = 24.0


def _reflect(value: float) -> float:
    span = FIELD_BOTTOM - FIELD_TOP
    folded = (value - FIELD_TOP) % (2.0 * span)
    if folded <= span:
        return FIELD_TOP + folded
    return FIELD_BOTTOM - (folded - span)


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self.previous_ball: tuple[float, float] | None = None
        self.velocity: tuple[float, float] | None = None

    def _track_ball(
        self,
        ball: tuple[float, float] | None,
    ) -> None:
        if ball is None:
            self.previous_ball = None
            self.velocity = None
            return

        if self.previous_ball is not None:
            dx = ball[0] - self.previous_ball[0]
            dy = ball[1] - self.previous_ball[1]
            plausible = (
                abs(dx) <= MAX_VELOCITY_JUMP
                and abs(dy) <= MAX_VELOCITY_JUMP
                and abs(dx) + abs(dy) > 0.25
            )
            if plausible:
                if self.velocity is None:
                    self.velocity = (dx, dy)
                else:
                    self.velocity = (
                        0.5 * (self.velocity[0] + dx),
                        0.5 * (self.velocity[1] + dy),
                    )
            else:
                self.velocity = None
        self.previous_ball = ball

    def act(self, observation: np.ndarray) -> int:
        ram = np.asarray(observation, dtype=np.uint8)
        if ram.shape != (128,):
            raise ValueError(f"Expected Pong RAM shape (128,), got {ram.shape}")

        ball = None
        if int(ram[54]) != 0:
            ball = (float(ram[49]) - 49.0, float(ram[54]) - 13.0)
        self_y = 0.972157 * float(ram[51]) - 2.553996
        opponent_y = 0.981619 * float(ram[50]) - 5.492890
        self._track_ball(ball)

        target_y = HOME_Y
        if ball is not None:
            ball_x, ball_y = ball
            if self.velocity is not None and self.velocity[0] > 0.05:
                velocity_x, velocity_y = self.velocity
                if ball_x <= PADDLE_X:
                    target_y = _reflect(
                        ball_y + velocity_y / velocity_x * (PADDLE_X - ball_x)
                    )
                else:
                    target_y = ball_y
                outgoing_sign = 1.0 if opponent_y < HOME_Y else -1.0
                target_y -= outgoing_sign * SPIN_OFFSET
            elif self.velocity is None and ball_x > 90.0:
                target_y = ball_y

        error = target_y - self_y
        if error < -DEADBAND:
            return MOVE_UP
        if error > DEADBAND:
            return MOVE_DOWN
        return NOOP
