"""Runnable seeded-random contract artifact, not the article score anchor."""

from __future__ import annotations

import numpy as np


class Policy:
    def __init__(
        self,
        *,
        action_count: int,
        env_id: str = "",
        repeat_index: int = 0,
        seed: int = 0,
        **_: object,
    ) -> None:
        self.action_count = action_count
        self.offset = (
            sum(ord(character) for character in env_id) + 1_000_003 * repeat_index
        )
        self.reset(seed=seed)

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed + self.offset)

    def act(self, observation, info=None) -> int:
        del observation, info
        return int(self.rng.integers(self.action_count))
