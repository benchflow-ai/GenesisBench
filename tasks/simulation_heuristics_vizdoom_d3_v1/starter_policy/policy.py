"""Weak sweep-and-fire starter for VizDoom D3 Battle."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _action(
    *,
    attack: float = 0.0,
    speed: float = 0.0,
    forward: float = 0.0,
    backward: float = 0.0,
    right: float = 0.0,
    left: float = 0.0,
    turn180: float = 0.0,
    turn: float = 0.0,
) -> np.ndarray:
    return np.asarray(
        [attack, speed, forward, backward, right, left, turn180, turn],
        dtype=np.float64,
    )


class Policy:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.direction = 1.0

    def act(
        self,
        frame: np.ndarray,
        variables: Mapping[str, float],
    ) -> np.ndarray:
        del frame
        ammo = float(variables["AMMO2"])
        phase = self.step % 120
        self.step += 1
        if phase == 90:
            self.direction *= -1.0
        if phase < 90:
            return _action(
                attack=1.0 if ammo > 0 else 0.0,
                speed=1.0,
                forward=1.0,
                turn=self.direction * 2.0,
            )
        return _action(
            attack=1.0 if ammo > 0 else 0.0,
            speed=1.0,
            turn=self.direction * 6.0,
        )

