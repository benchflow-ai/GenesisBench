from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from genesisbench.ant import DynamicsVariant, evaluate_ant_policy
from scripts.validate_tasks import validate_task


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "tasks" / "simulation_heuristics_ant_v1"
STARTER = TASK_DIR / "starter_policy" / "policy.py"
ORACLE = TASK_DIR / "oracle" / "policy.py"
REFERENCE = TASK_DIR / "verifier" / "anchor_policies" / "reference_policy.py"
HIDDEN_EVALUATOR = TASK_DIR / "verifier" / "evaluate_hidden.py"
PROVENANCE = TASK_DIR / "evidence" / "source_provenance.json"


def test_starter_policy_short_smoke() -> None:
    result = evaluate_ant_policy(
        STARTER,
        seeds=(0,),
        max_steps=10,
    )

    assert len(result.episodes) == 1
    assert result.episodes[0].length == 10
    assert result.invalid_episode_rate == 0.0
    assert result.mean_return > 0.0


def test_dynamics_variant_short_smoke() -> None:
    result = evaluate_ant_policy(
        STARTER,
        seeds=(1,),
        max_steps=5,
        variants=(
            DynamicsVariant(
                name="test_variant",
                density_scale=1.05,
                friction_scale=0.95,
                damping_scale=1.05,
                actuator_scale=0.95,
            ),
        ),
    )

    assert result.episodes[0].variant == "test_variant"
    assert result.invalid_episode_rate == 0.0


def test_json_uses_public_return_key() -> None:
    result = evaluate_ant_policy(
        STARTER,
        seeds=(2,),
        max_steps=2,
    )

    episode = result.to_dict()["episodes"][0]
    assert "return" in episode
    assert "return_" not in episode


def test_article_reference_can_plan_from_copied_model() -> None:
    result = evaluate_ant_policy(
        REFERENCE,
        seeds=(0,),
        max_steps=1,
    )

    assert result.episodes[0].length == 1
    assert result.invalid_episode_rate == 0.0
    assert result.mean_action_latency_ms > 0.0


def test_anchor_normalization_is_platform_local(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
version = "1.0"

[evaluation]
max_steps = 1
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
density_scale = 1.02
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
    assert results["starter"]["unique_policy_evaluations"] == 2
    assert results["reference"]["unique_policy_evaluations"] == 2


def test_task_package_validates() -> None:
    assert (
        validate_task(
            TASK_DIR,
            runtime_source=REPO_ROOT / "src" / "genesisbench",
        )
        == []
    )


def test_article_provenance_is_machine_readable() -> None:
    provenance = json.loads(PROVENANCE.read_text())

    assert (
        provenance["source"]["revision"] == "3555c2956c257d49a5015b782cbe485b14fd659e"
    )
    assert provenance["article_reproduction"]["seeds"] == [0, 1, 2, 3, 4]
    assert provenance["article_reproduction"]["reported_mean_return"] == 6005.521
    assert provenance["genesisbench_adaptation"]["source_and_gymnasium_xml_parse_equal"]
    assert provenance["local_genesisbench_oracle_on_gymnasium"][
        "matches_local_source_policy"
    ]


@pytest.mark.skipif(
    os.environ.get("GENESISBENCH_RUN_SLOW_ANT_MPC") != "1",
    reason="set GENESISBENCH_RUN_SLOW_ANT_MPC=1 for the full MPC reproduction",
)
def test_article_mpc_full_reproduction() -> None:
    result = evaluate_ant_policy(
        ORACLE,
        seeds=range(5),
        max_steps=1000,
    )

    assert [episode.length for episode in result.episodes] == [1000] * 5
    assert result.invalid_episode_rate == 0.0
    assert result.mean_return == pytest.approx(6005.521, abs=200.0)
    assert result.min_return == pytest.approx(5776.805, abs=100.0)
    assert result.max_return == pytest.approx(6146.208, abs=100.0)
