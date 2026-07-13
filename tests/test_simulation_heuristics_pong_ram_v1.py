from __future__ import annotations

from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from genesisbench.pong import PongVariant, evaluate_pong_policy


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "tasks" / "simulation_heuristics_pong_ram_v1"
STARTER = TASK_DIR / "starter_policy" / "policy.py"
REFERENCE = TASK_DIR / "oracle" / "policy.py"
HIDDEN_EVALUATOR = TASK_DIR / "verifier" / "evaluate_hidden.py"


class ScriptedPongEnv:
    def __init__(self) -> None:
        self.action_space = SimpleNamespace(n=6)
        self.actions: list[int] = []
        self.step_index = 0

    @staticmethod
    def _info() -> dict[str, np.ndarray]:
        ram = np.zeros((1, 128), dtype=np.uint8)
        ram[0, 49] = 120
        ram[0, 50] = 100
        ram[0, 51] = 100
        ram[0, 54] = 100
        return {"ram": ram}

    def reset(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        self.step_index = 0
        return np.zeros((1, 1), dtype=np.uint8), self._info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, np.ndarray],
    ]:
        self.actions.append(int(action[0]))
        self.step_index += 1
        return (
            np.zeros((1, 1), dtype=np.uint8),
            np.asarray([1.0]),
            np.asarray([False]),
            np.asarray([self.step_index == 2]),
            self._info(),
        )

    def close(self) -> None:
        pass


def test_policy_is_scored_from_ram_observations(tmp_path: Path) -> None:
    policy = tmp_path / "policy.py"
    policy.write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "class Policy:",
                "    def reset(self, seed=0):",
                "        self.seed = seed",
                "",
                "    def act(self, observation):",
                "        ram = np.asarray(observation)",
                "        assert ram.shape == (128,)",
                "        assert ram.dtype == np.uint8",
                "        return np.int64(0)",
                "",
            ]
        )
    )
    environments: list[ScriptedPongEnv] = []

    def make_env(seed: int, variant: PongVariant) -> ScriptedPongEnv:
        assert seed == 7
        assert variant.name == "test"
        environment = ScriptedPongEnv()
        environments.append(environment)
        return environment

    result = evaluate_pong_policy(
        policy,
        seeds=(7,),
        max_steps=10,
        variants=(PongVariant(name="test"),),
        environment_factory=make_env,
    )

    assert result.mean_score == 2.0
    assert result.invalid_episode_rate == 0.0
    assert result.episodes[0].points_for == 2
    assert result.episodes[0].points_against == 0
    assert result.episodes[0].length == 2
    assert environments[0].actions == [0, 0]


def test_invalid_discrete_action_receives_failure_score(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "policy.py"
    policy.write_text(
        "\n".join(
            [
                "class Policy:",
                "    def act(self, observation):",
                "        return 99",
                "",
            ]
        )
    )

    result = evaluate_pong_policy(
        policy,
        seeds=(3,),
        variants=(PongVariant(name="invalid"),),
        environment_factory=lambda seed, variant: ScriptedPongEnv(),
    )

    assert result.mean_score == -21.0
    assert result.invalid_episode_rate == 1.0
    assert result.episodes[0].points_for == 0
    assert result.episodes[0].points_against == 21
    assert "must be in [0, 5]" in (result.episodes[0].policy_error or "")


@pytest.mark.skipif(
    find_spec("envpool") is None,
    reason="EnvPool is installed in the packaged Pong task environment.",
)
def test_reference_reproduces_article_score_21() -> None:
    result = evaluate_pong_policy(
        REFERENCE,
        seeds=(0,),
        max_steps=27_000,
        variants=(PongVariant(name="article", noop_max=1),),
    )

    assert result.mean_score == 21.0
    assert result.target_score_rate == 1.0
    assert result.invalid_episode_rate == 0.0


def test_native_task_package_validates_and_hides_trusted_files(
    tmp_path: Path,
) -> None:
    from scripts.prepare_task import prepare_task
    from scripts.validate_tasks import validate_task

    runtime_source = REPO_ROOT / "src" / "genesisbench"
    assert (
        validate_task(
            TASK_DIR,
            runtime_source=runtime_source,
        )
        == []
    )

    prepared = prepare_task(
        TASK_DIR.name,
        tmp_path / TASK_DIR.name,
        tasks_root=TASK_DIR.parent,
        runtime_source=runtime_source,
    )

    assert not (prepared / "verifier").exists()
    assert not (prepared / "oracle").exists()
    assert not (prepared / "evidence").exists()
    assert (prepared / "final_policy" / "policy.py").read_bytes() == (
        STARTER.read_bytes()
    )
    assert (prepared / "_runtime" / "genesisbench" / "pong.py").is_file()


@pytest.mark.skipif(
    find_spec("envpool") is None,
    reason="EnvPool is installed in the packaged Pong task environment.",
)
def test_anchor_normalization_is_platform_local(tmp_path: Path) -> None:
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
    assert results["reference"]["score"] == 21
    assert results["reference"]["reference_score"] == 21


def test_hidden_suite_uses_multiple_seeds_and_reset_configs() -> None:
    config = tomllib.loads((TASK_DIR / "verifier" / "config.toml").read_text())
    evaluation = config["evaluation"]
    suites = evaluation["suites"]
    variants = evaluation["variants"]

    seeds = [int(seed) for suite in suites for seed in suite["seeds"]]
    assert len(seeds) >= 6
    assert len(set(seeds)) == len(seeds)
    assert {suite["variant"] for suite in suites} == {
        "article_nominal",
        "randomized_reset",
    }
    assert {int(variant["noop_max"]) for variant in variants} == {1, 30}
    assert all(int(variant["frame_skip"]) == 1 for variant in variants)
    assert all(
        float(variant["repeat_action_probability"]) == 0.0 for variant in variants
    )
