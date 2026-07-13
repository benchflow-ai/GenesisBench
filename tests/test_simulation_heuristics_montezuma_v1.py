from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
import pytest

from genesisbench.montezuma import MontezumaVariant, evaluate_montezuma_policy
from scripts.validate_tasks import validate_task


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "tasks" / "simulation_heuristics_montezuma_v1"
STARTER = TASK_DIR / "starter_policy" / "policy.py"
REFERENCE = TASK_DIR / "oracle" / "policy.py"
REFERENCE_DATA = TASK_DIR / "oracle" / "reference_trajectory.npz"
HIDDEN_EVALUATOR = TASK_DIR / "verifier" / "evaluate_hidden.py"


class _ImageActionEnv:
    def __init__(self) -> None:
        self.state = 0
        self.closed = False

    def _observation(self) -> np.ndarray:
        return np.full((1, 3, 4, 5), self.state, dtype=np.uint8)

    def reset(self) -> tuple[np.ndarray, dict[str, object]]:
        self.state = 0
        return self._observation(), {"private_state": "not for the policy"}

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        action = int(np.asarray(actions).reshape(-1)[0])
        if action == self.state:
            self.state += 1
        reward = 400.0 if self.state == 3 else 0.0
        terminated = self.state == 3
        return (
            self._observation(),
            np.asarray([reward], dtype=np.float32),
            np.asarray([terminated]),
            np.asarray([False]),
            {"private_state": self.state},
        )

    def close(self) -> None:
        self.closed = True


class _RecoveryEnv:
    def __init__(self) -> None:
        self.state = 0

    def _observation(self) -> np.ndarray:
        return np.full((1, 3, 4, 5), self.state, dtype=np.uint8)

    def reset(self) -> tuple[np.ndarray, dict[str, object]]:
        self.state = 0
        return self._observation(), {}

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        action = int(np.asarray(actions).reshape(-1)[0])
        failed = action != self.state
        if not failed:
            self.state += 1
        reward = 400.0 if self.state == 6 else 0.0
        terminated = failed or self.state == 6
        return (
            self._observation(),
            np.asarray([reward], dtype=np.float32),
            np.asarray([terminated]),
            np.asarray([False]),
            {},
        )

    def close(self) -> None:
        return None


def test_policy_receives_native_rgb_only_and_reaches_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy = tmp_path / "policy.py"
    policy.write_text(
        """
import numpy as np


class Policy:
    def reset(self, seed=0):
        self.seed = seed

    def act(self, observation):
        frame = np.asarray(observation)
        assert frame.shape == (3, 4, 5)
        assert frame.dtype == np.uint8
        return int(frame[0, 0, 0])
""".lstrip()
    )
    environments: list[_ImageActionEnv] = []

    def make_env(*, seed: int, max_steps: int) -> _ImageActionEnv:
        assert seed == 17
        assert max_steps == 8
        environment = _ImageActionEnv()
        environments.append(environment)
        return environment

    monkeypatch.setattr(
        "genesisbench.montezuma._make_montezuma_env",
        make_env,
    )

    result = evaluate_montezuma_policy(
        policy,
        seeds=(17,),
        variants=(MontezumaVariant(),),
        max_steps=8,
    )

    assert result.mean_return == 400.0
    assert result.target_success_rate == 1.0
    assert result.invalid_episode_rate == 0.0
    assert result.episodes[0].policy_steps == 3
    assert environments[0].closed


def test_recovery_variant_requires_reentry_from_the_observed_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_policy = tmp_path / "image_policy.py"
    image_policy.write_text(
        """
import numpy as np


class Policy:
    def reset(self, seed=0):
        pass

    def act(self, observation):
        return int(np.asarray(observation)[0, 0, 0])
""".lstrip()
    )
    open_loop_policy = tmp_path / "open_loop_policy.py"
    open_loop_policy.write_text(
        """
class Policy:
    def reset(self, seed=0):
        self.step = 0

    def act(self, observation):
        action = self.step
        self.step += 1
        return action
""".lstrip()
    )
    monkeypatch.setattr(
        "genesisbench.montezuma._make_montezuma_env",
        lambda **_: _RecoveryEnv(),
    )
    variant = MontezumaVariant(
        name="checkpoint_reentry",
        bootstrap_steps=3,
    )

    recovered = evaluate_montezuma_policy(
        image_policy,
        seeds=(5,),
        variants=(variant,),
        bootstrap_policy_path=image_policy,
        max_steps=12,
    )
    copied_route = evaluate_montezuma_policy(
        open_loop_policy,
        seeds=(5,),
        variants=(variant,),
        bootstrap_policy_path=image_policy,
        max_steps=12,
    )

    assert recovered.mean_return == 400.0
    assert recovered.recovery_success_rate == 1.0
    assert recovered.episodes[0].policy_steps == 3
    assert copied_route.mean_return == 0.0
    assert copied_route.recovery_success_rate == 0.0


def test_invalid_action_receives_fixed_failure_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy = tmp_path / "policy.py"
    policy.write_text(
        """
class Policy:
    def act(self, observation):
        return 18
""".lstrip()
    )
    monkeypatch.setattr(
        "genesisbench.montezuma._make_montezuma_env",
        lambda **_: _ImageActionEnv(),
    )

    result = evaluate_montezuma_policy(
        policy,
        seeds=(0,),
        max_steps=4,
        failure_score=-25.0,
    )

    episode = result.episodes[0]
    assert episode.return_ == -25.0
    assert episode.invalid_action
    assert episode.policy_error is not None
    assert result.invalid_episode_rate == 1.0
    assert "return" in result.to_dict()["episodes"][0]
    assert "return_" not in result.to_dict()["episodes"][0]


def test_native_task_package_and_history_validate() -> None:
    assert (
        validate_task(
            TASK_DIR,
            runtime_source=REPO_ROOT / "src" / "genesisbench",
        )
        == []
    )

    task_text = (TASK_DIR / "task.md").read_text()
    provenance = (TASK_DIR / "task_context" / "provenance.md").read_text()
    assert "final_policy/policy.py" in task_text
    assert "native RGB" in task_text
    assert "`72` to `28`" in provenance
    assert "`86`" in provenance
    assert "`1769`" in provenance
    assert "mostly" in provenance and "open-loop" in provenance

    config = tomllib.loads((TASK_DIR / "verifier" / "config.toml").read_text())
    weights = {
        suite["name"]: suite["weight"] for suite in config["evaluation"]["suites"]
    }
    assert weights["recovery"] > 0.5
    assert sum(weight for name, weight in weights.items() if name != "recovery") < 0.5

    with np.load(REFERENCE_DATA, allow_pickle=False) as trajectory:
        assert set(trajectory.files) == {"actions", "hashes", "features"}
        assert trajectory["actions"].shape == (1769,)


def test_real_reference_reproduces_400_and_blocks_plain_replay(
    tmp_path: Path,
) -> None:
    pytest.importorskip("envpool")
    canonical = evaluate_montezuma_policy(
        REFERENCE,
        seeds=(10001,),
        max_steps=2000,
    )
    assert canonical.mean_return == 400.0
    assert canonical.episodes[0].length == 1769

    open_loop = tmp_path / "policy.py"
    open_loop.write_text(
        f"""
from pathlib import Path
import numpy as np

with np.load(Path({str(REFERENCE_DATA)!r}), allow_pickle=False) as data:
    ACTIONS = np.asarray(data["actions"], dtype=np.int64)


class Policy:
    def reset(self, seed=0):
        self.step = 0

    def act(self, observation):
        del observation
        if self.step >= len(ACTIONS):
            return 0
        action = int(ACTIONS[self.step])
        self.step += 1
        return action
""".lstrip()
    )
    recovery_variants = (
        MontezumaVariant("recover_512_noop_4", 512, 4),
        MontezumaVariant("recover_512_noop_8", 512, 8),
        MontezumaVariant("recover_768_noop_2", 768, 2),
        MontezumaVariant("recover_900", 900, 0),
    )
    recovered = evaluate_montezuma_policy(
        REFERENCE,
        seeds=(10001,),
        variants=recovery_variants,
        bootstrap_policy_path=REFERENCE,
    )
    copied_route = evaluate_montezuma_policy(
        open_loop,
        seeds=(10001,),
        variants=recovery_variants,
        bootstrap_policy_path=REFERENCE,
    )

    assert recovered.capped_mean_score == 400.0
    assert recovered.recovery_success_rate == 1.0
    assert copied_route.capped_mean_score == 0.0
    assert copied_route.recovery_success_rate == 0.0


def test_anchor_normalization_is_platform_local(tmp_path: Path) -> None:
    pytest.importorskip("envpool")
    results = {}
    for name, policy in (("starter", STARTER), ("reference", REFERENCE)):
        output = tmp_path / f"{name}.json"
        subprocess.run(
            [
                sys.executable,
                str(HIDDEN_EVALUATOR),
                str(policy),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        results[name] = json.loads(output.read_text())

    assert results["starter"]["normalized_score"] == 0
    assert results["reference"]["normalized_score"] == 100
