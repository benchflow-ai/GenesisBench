"""Runnable zero-interaction baseline for every Atari57 game and input mode."""

from __future__ import annotations


class Policy:
    def __init__(
        self,
        *,
        action_count: int,
        seed: int = 0,
        **_: object,
    ) -> None:
        self.action_count = action_count
        self.reset(seed=seed)

    def reset(self, seed: int = 0) -> None:
        self.seed = seed

    def act(self, observation, info=None) -> int:
        del observation, info
        return 0
