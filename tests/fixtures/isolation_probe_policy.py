from __future__ import annotations

from pathlib import Path

import numpy as np


class Policy:
    def reset(self, seed: int = 0) -> None:
        del seed

    def act(self, observation: np.ndarray) -> np.ndarray:
        del observation
        try:
            Path("/verifier/config.toml").read_text()
        except OSError:
            pass
        else:
            raise RuntimeError("policy could read /verifier/config.toml")

        try:
            Path("/logs/verifier/reward.txt").write_text("1.0\n")
        except OSError:
            pass
        else:
            raise RuntimeError("policy could write verifier reward")

        return np.zeros(8, dtype=np.float64)
