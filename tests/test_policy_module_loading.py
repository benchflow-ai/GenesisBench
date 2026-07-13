from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import pytest

from genesisbench.ant import _load_policy_module as load_ant_policy
from genesisbench.atari57 import _load_policy_module as load_atari57_policy
from genesisbench.breakout import _load_policy_module as load_breakout_policy
from genesisbench.halfcheetah import _load_policy_module as load_halfcheetah_policy
from genesisbench.montezuma import _load_policy_module as load_montezuma_policy
from genesisbench.pong import _load_policy_module as load_pong_policy
from genesisbench.vizdoom import _load_policy_module as load_vizdoom_policy


PolicyLoader = Callable[[Path], object]


@pytest.mark.parametrize(
    "loader",
    (
        load_ant_policy,
        load_atari57_policy,
        load_breakout_policy,
        load_halfcheetah_policy,
        load_montezuma_policy,
        load_pong_policy,
        load_vizdoom_policy,
    ),
)
def test_policy_loader_supports_postponed_dataclass_annotations(
    tmp_path: Path,
    loader: PolicyLoader,
) -> None:
    policy_path = tmp_path / f"{loader.__module__.replace('.', '_')}.py"
    policy_path.write_text(
        """
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Policy:
    gain: float = 1.0
""".lstrip()
    )

    module = loader(policy_path)
    try:
        assert sys.modules[module.__name__] is module
        assert module.Policy().gain == 1.0
    finally:
        sys.modules.pop(module.__name__, None)
