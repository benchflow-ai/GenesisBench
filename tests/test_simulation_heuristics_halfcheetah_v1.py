from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from genesisbench.halfcheetah import (
    DynamicsVariant,
    evaluate_halfcheetah_policy,
)
from scripts.prepare_task import prepare_task
from scripts.validate_tasks import validate_task


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "tasks" / "simulation_heuristics_halfcheetah_v1"
STARTER = TASK_DIR / "starter_policy" / "policy.py"
REFERENCE = TASK_DIR / "verifier" / "anchor_policies" / "reference_policy.py"
HIDDEN_EVALUATOR = TASK_DIR / "verifier" / "evaluate_hidden.py"
ARTICLE_EVIDENCE = TASK_DIR / "evidence" / "article_reproduction.json"


def test_starter_policy_short_smoke() -> None:
    result = evaluate_halfcheetah_policy(
        STARTER,
        seeds=(100,),
        max_steps=10,
    )

    assert len(result.episodes) == 1
    assert result.episodes[0].length == 10
    assert result.invalid_episode_rate == 0.0
    assert result.mean_return > 0.0


def test_dynamics_variant_short_smoke() -> None:
    result = evaluate_halfcheetah_policy(
        STARTER,
        seeds=(101,),
        max_steps=5,
        variants=(
            DynamicsVariant(
                name="test_variant",
                mass_scale=1.05,
                friction_scale=0.95,
                damping_scale=1.05,
                actuator_scale=0.95,
            ),
        ),
    )

    assert result.episodes[0].variant == "test_variant"
    assert result.invalid_episode_rate == 0.0


def test_json_uses_public_return_key() -> None:
    result = evaluate_halfcheetah_policy(
        STARTER,
        seeds=(102,),
        max_steps=2,
    )

    episode = result.to_dict()["episodes"][0]
    assert "return" in episode
    assert "return_" not in episode


def test_reference_policy_can_plan_from_observations() -> None:
    result = evaluate_halfcheetah_policy(
        REFERENCE,
        seeds=(100,),
        max_steps=1,
    )

    assert result.episodes[0].length == 1
    assert result.invalid_episode_rate == 0.0
    assert result.mean_action_latency_ms > 0.0


def test_task_package_validates() -> None:
    assert (
        validate_task(
            TASK_DIR,
            runtime_source=REPO_ROOT / "src" / "genesisbench",
        )
        == []
    )


def test_prepared_workspace_seeds_final_policy(tmp_path: Path) -> None:
    prepared = prepare_task(
        TASK_DIR.name,
        tmp_path / TASK_DIR.name,
        tasks_root=TASK_DIR.parent,
        runtime_source=REPO_ROOT / "src" / "genesisbench",
    )

    assert not (prepared / "verifier").exists()
    assert not (prepared / "oracle").exists()
    assert not (prepared / "evidence").exists()
    assert (
        prepared / "final_policy" / "policy.py"
    ).read_bytes() == STARTER.read_bytes()


def test_anchor_normalization_is_platform_local(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
version = "1.0"

[evaluation]
max_steps = 2
failure_return = -1000.0

[[evaluation.suites]]
name = "hidden_nominal"
weight = 0.7
seeds = [701]

[[evaluation.suites]]
name = "hidden_robustness"
weight = 0.3
seeds = [703]

[[evaluation.variants]]
name = "test_variant"
mass_scale = 1.02
friction_scale = 0.98
damping_scale = 1.01
actuator_scale = 0.99
""".lstrip()
    )

    results = {}
    for name, policy in (("starter", STARTER), ("reference", REFERENCE)):
        output = tmp_path / f"{name}.json"
        subprocess.run(
            [
                sys.executable,
                str(HIDDEN_EVALUATOR),
                str(policy),
                "--config",
                str(config),
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


def test_article_five_seed_reproduction_is_recorded() -> None:
    evidence = json.loads(ARTICLE_EVIDENCE.read_text())

    assert evidence["article"]["seeds"] == [100, 101, 102, 103, 104]
    assert evidence["observed"]["mean_return"] == 11836.693449819431
    assert evidence["observed"]["min_return"] == 11735.02927325886
    assert evidence["observed"]["max_return"] == 12041.189857475818
