from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib

import numpy as np
import pytest

from genesisbench.atari57 import (
    ATARI57_GAMES,
    ATARI_SETTINGS,
    Atari57ArtifactError,
    AtariEpisode,
    FRAME_BUDGET_PER_SEARCH,
    HNSReference,
    OBSERVATION_MODES,
    aggregate_atari57_episodes,
    evaluate_atari57_artifact,
    expected_search_trajectories,
    load_hns_references,
    load_atari57_artifact,
)
from scripts.validate_tasks import validate_task


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "tasks" / "simulation_heuristics_atari57_v1"
STARTER_ARTIFACT = TASK_DIR / "starter_artifact"
REFERENCE_ARTIFACT = TASK_DIR / "oracle" / "reference_artifact"
HIDDEN_EVALUATOR = TASK_DIR / "verifier" / "evaluate_hidden.py"
PUBLIC_EVALUATOR = TASK_DIR / "evaluate.py"
RUNTIME_SOURCE = REPO_ROOT / "src" / "genesisbench"


def test_article_protocol_expands_to_342_search_trajectories() -> None:
    trajectories = expected_search_trajectories()

    assert len(ATARI57_GAMES) == 57
    assert len(set(ATARI57_GAMES)) == 57
    assert OBSERVATION_MODES == ("ram", "native_obs")
    assert len(trajectories) == 342
    assert (
        len({(item.env_id, item.obs_mode, item.repeat_index) for item in trajectories})
        == 342
    )
    assert {item.repeat_index for item in trajectories} == {0, 1, 2}
    assert sum(item.frame_budget for item in trajectories) == (
        342 * FRAME_BUDGET_PER_SEARCH
    )


def test_starter_artifact_is_complete_and_honest_about_unused_budget() -> None:
    artifact = load_atari57_artifact(STARTER_ARTIFACT)

    assert len(artifact.policies) == 342
    assert {
        (policy.env_id, policy.obs_mode, policy.repeat_index)
        for policy in artifact.policies
    } == {
        (env_id, obs_mode, repeat_index)
        for env_id in ATARI57_GAMES
        for obs_mode in OBSERVATION_MODES
        for repeat_index in range(3)
    }
    assert all(policy.module_path.is_file() for policy in artifact.policies)

    budget = artifact.interaction_budget
    assert budget.planned_trajectories == 342
    assert budget.completed_trajectories == 0
    assert budget.counted_env_steps == 0
    assert budget.target_env_steps == 6_840_000_000


def test_completed_search_record_requires_article_evidence(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    (artifact_root / "manifest.json").write_text(
        (STARTER_ARTIFACT / "manifest.json").read_text()
    )
    (artifact_root / "policy.py").write_text(
        (STARTER_ARTIFACT / "policy.py").read_text()
    )
    record = {
        "env_id": "Breakout-v5",
        "obs_mode": "native_obs",
        "repeat_index": 0,
        "cumulative_env_steps": FRAME_BUDGET_PER_SEARCH,
        "cumulative_episodes": 100,
        "status": "complete",
        "evidence_path": "searches/breakout/native_obs/repeat_0",
    }
    ledger_path = artifact_root / "interaction_ledger.json"
    ledger_path.write_text(json.dumps({"schema_version": "1.0", "records": [record]}))

    with pytest.raises(
        Atari57ArtifactError,
        match="complete search evidence is missing",
    ):
        load_atari57_artifact(artifact_root)

    evidence = artifact_root / record["evidence_path"]
    evidence.mkdir(parents=True)
    for filename in ("policy.py", "trials.jsonl", "summary.csv", "README.md"):
        (evidence / filename).write_text("fixture\n")
    (evidence / "sample_efficiency.png").write_bytes(b"fixture")

    artifact = load_atari57_artifact(artifact_root)
    assert artifact.interaction_budget.completed_trajectories == 1
    assert artifact.interaction_records[0].evidence_path == evidence


def test_repeat_specific_policy_override_is_evaluated_for_that_repeat(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    manifest = json.loads((STARTER_ARTIFACT / "manifest.json").read_text())
    manifest["policies"] = [
        {
            "env_id": "Breakout-v5",
            "obs_mode": "native_obs",
            "repeat_index": 1,
            "module": "repeat_one.py",
        }
    ]
    (artifact_root / "manifest.json").write_text(json.dumps(manifest))
    (artifact_root / "interaction_ledger.json").write_text(
        '{"schema_version": "1.0", "records": []}\n'
    )
    (artifact_root / "policy.py").write_text(
        """
class Policy:
    def act(self, observation, info=None):
        return 0
""".strip()
        + "\n"
    )
    (artifact_root / "repeat_one.py").write_text(
        """
class Policy:
    def __init__(self, *, repeat_index, **kwargs):
        assert repeat_index == 1

    def act(self, observation, info=None):
        return 1
""".strip()
        + "\n"
    )

    class FakeActionSpace:
        n = 2

    class FakeEnv:
        action_space = FakeActionSpace()

        def reset(self):
            return np.asarray([[0.0]]), {}

        def step(self, action):
            return (
                np.asarray([[0.0]]),
                np.asarray([float(action[0])]),
                np.asarray([True]),
                {},
            )

        def close(self) -> None:
            return None

    artifact = load_atari57_artifact(artifact_root)
    repeat_one = next(
        policy
        for policy in artifact.policies
        if (
            policy.env_id,
            policy.obs_mode,
            policy.repeat_index,
        )
        == ("Breakout-v5", "native_obs", 1)
    )
    assert repeat_one.module_path.name == "repeat_one.py"

    result = evaluate_atari57_artifact(
        artifact,
        games=("Breakout-v5",),
        obs_modes=("native_obs",),
        seeds=(10, 11, 12),
        max_steps=1,
        hns_references={
            "Breakout-v5": HNSReference(
                env_id="Breakout-v5",
                known_best_score=864.0,
                random_score=0.0,
                human_score=1.0,
            )
        },
        env_factory=lambda env_id, seed, settings: FakeEnv(),
    )

    assert [episode.return_ for episode in result.episodes] == [0.0, 1.0, 0.0]
    assert result.score == pytest.approx(1.0 / 3.0)


def test_hns_aggregation_matches_article_best_input_semantics() -> None:
    values = {
        "Alien-v5": {
            "ram": (0.1, 0.2, 0.3),
            "native_obs": (0.4, 0.5, 0.6),
        },
        "Breakout-v5": {
            "ram": (1.0, 2.0, 3.0),
            "native_obs": (4.0, 0.0, 0.0),
        },
        "Pong-v5": {
            "ram": (-0.2, 0.0, 0.2),
            "native_obs": (0.1, 0.1, 0.1),
        },
    }
    episodes = tuple(
        AtariEpisode(
            env_id=env_id,
            obs_mode=obs_mode,
            repeat_index=repeat_index,
            seed=repeat_index,
            return_=hns,
            hns=hns,
            length=1,
            terminated=True,
            truncated=False,
            invalid_action=False,
            policy_error=None,
            mean_action_latency_ms=0.0,
        )
        for env_id, modes in values.items()
        for obs_mode, repeats in modes.items()
        for repeat_index, hns in enumerate(repeats)
    )

    result = aggregate_atari57_episodes(episodes)

    assert result.evaluation_trajectories == 18
    assert result.games_evaluated == 3
    assert result.per_game["Alien-v5"]["best_input_mean_hns"] == 0.5
    assert result.per_game["Alien-v5"]["modes"]["ram"]["repeat_count"] == 3
    assert result.per_game["Breakout-v5"]["best_input_mean_hns"] == 2.0
    assert result.per_game["Breakout-v5"]["best_single_run_hns"] == 4.0
    assert result.score == 0.5
    assert result.median_best_single_run_hns == 0.6


def test_public_hns_table_covers_atari57_and_uses_random_human_anchors() -> None:
    references = load_hns_references(TASK_DIR / "task_context" / "atari57_games.csv")

    assert len(references) == 57
    assert set(references) == set(ATARI57_GAMES)

    breakout = references["Breakout-v5"]
    assert breakout.random_score == 1.7
    assert breakout.human_score == 30.5
    assert breakout.normalize(1.7) == 0.0
    assert breakout.normalize(30.5) == 1.0
    assert round(breakout.normalize(864.0), 12) == round(
        29.94097222222222,
        12,
    )


def test_evaluator_enforces_observation_mode_boundary_and_fixed_settings(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    manifest = json.loads((STARTER_ARTIFACT / "manifest.json").read_text())
    (artifact_root / "manifest.json").write_text(json.dumps(manifest))
    (artifact_root / "interaction_ledger.json").write_text(
        '{"schema_version": "1.0", "records": []}\n'
    )
    (artifact_root / "policy.py").write_text(
        """
class Policy:
    def __init__(self, *, obs_mode, action_count, **kwargs):
        self.obs_mode = obs_mode
        self.action_count = action_count

    def reset(self, seed=0):
        self.seed = seed

    def act(self, observation, info=None):
        if self.obs_mode == "native_obs" and info is not None:
            raise RuntimeError("native_obs leaked info")
        if self.obs_mode == "ram":
            if set(info) != {"ram"}:
                raise RuntimeError("RAM policy received hidden info")
            if int(info["ram"][0]) != 9:
                raise RuntimeError("RAM payload was not unbatched")
        return 1
""".strip()
        + "\n"
    )

    factory_calls: list[tuple[str, int, dict[str, object]]] = []

    class FakeActionSpace:
        n = 3

    class FakeEnv:
        action_space = FakeActionSpace()

        def __init__(self) -> None:
            self.steps = 0

        def reset(self):
            return (
                np.asarray([[1.0, 2.0]]),
                {
                    "ram": np.asarray([[9, 8]], dtype=np.uint8),
                    "secret": np.asarray([123]),
                },
            )

        def step(self, action):
            assert action.tolist() == [1]
            self.steps += 1
            return (
                np.asarray([[3.0, 4.0]]),
                np.asarray([1.0]),
                np.asarray([self.steps == 2]),
                {
                    "ram": np.asarray([[9, 7]], dtype=np.uint8),
                    "secret": np.asarray([456]),
                },
            )

        def close(self) -> None:
            return None

    def fake_env_factory(
        env_id: str,
        seed: int,
        settings: dict[str, object],
    ) -> FakeEnv:
        factory_calls.append((env_id, seed, settings))
        return FakeEnv()

    result = evaluate_atari57_artifact(
        artifact_root,
        games=("Breakout-v5",),
        obs_modes=OBSERVATION_MODES,
        seeds=(7,),
        max_steps=5,
        hns_references={
            "Breakout-v5": HNSReference(
                env_id="Breakout-v5",
                known_best_score=864.0,
                random_score=0.0,
                human_score=2.0,
            )
        },
        env_factory=fake_env_factory,
    )

    assert result.score == 1.0
    assert result.evaluation_trajectories == 2
    assert result.counted_evaluation_steps == 4
    assert result.invalid_episode_rate == 0.0
    assert factory_calls == [
        ("Breakout-v5", 7, ATARI_SETTINGS),
        ("Breakout-v5", 7, ATARI_SETTINGS),
    ]


def test_invalid_policy_call_counts_zero_environment_steps(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    (artifact_root / "manifest.json").write_text(
        (STARTER_ARTIFACT / "manifest.json").read_text()
    )
    (artifact_root / "interaction_ledger.json").write_text(
        '{"schema_version": "1.0", "records": []}\n'
    )
    (artifact_root / "policy.py").write_text(
        """
class Policy:
    def act(self, observation, info=None):
        return 99
""".strip()
        + "\n"
    )

    class FakeActionSpace:
        n = 2

    class FakeEnv:
        action_space = FakeActionSpace()

        def reset(self):
            return np.asarray([[0.0]]), {}

        def step(self, action):
            raise AssertionError("invalid policy must fail before env.step")

        def close(self) -> None:
            return None

    result = evaluate_atari57_artifact(
        artifact_root,
        games=("Breakout-v5",),
        obs_modes=("native_obs",),
        seeds=(1,),
        max_steps=5,
        hns_references={
            "Breakout-v5": HNSReference(
                env_id="Breakout-v5",
                known_best_score=864.0,
                random_score=0.0,
                human_score=1.0,
            )
        },
        env_factory=lambda env_id, seed, settings: FakeEnv(),
    )

    assert result.counted_evaluation_steps == 0
    assert result.invalid_episode_rate == 1.0
    assert result.episodes[0].length == 0
    assert result.episodes[0].hns == 0.0


def test_numeric_anchors_map_natural_origin_and_article_reference_to_0_100(
    tmp_path: Path,
) -> None:
    anchors_path = TASK_DIR / "verifier" / "anchors.json"
    anchors = json.loads(anchors_path.read_text())
    assert anchors["starter_policy"]["score"] == 0.0
    assert anchors["reference_policy"]["score"] == 0.8283015254994576
    assert "path" not in anchors["starter_policy"]
    assert "path" not in anchors["reference_policy"]
    oracle = load_atari57_artifact(REFERENCE_ARTIFACT)
    assert len(oracle.policies) == 342
    assert oracle.interaction_budget.completed_trajectories == 0

    output = tmp_path / "numeric.json"
    subprocess.run(
        [
            sys.executable,
            str(HIDDEN_EVALUATOR),
            str(STARTER_ARTIFACT),
            "--config",
            str(TASK_DIR / "verifier" / "config_smoke.toml"),
            "--anchors",
            str(anchors_path),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text())
    assert payload["starter_score"] == 0.0
    assert payload["reference_score"] == 0.8283015254994576
    assert (
        100.0
        * payload["reference_score"]
        / (payload["reference_score"] - payload["starter_score"])
        == 100.0
    )
    assert payload["evaluation"]["evaluation_trajectories"] == 18


def test_incomplete_artifact_is_disqualified_before_full_envpool_evaluation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "incomplete.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(HIDDEN_EVALUATOR),
            str(STARTER_ARTIFACT),
            "--config",
            str(TASK_DIR / "verifier" / "config.toml"),
            "--anchors",
            str(TASK_DIR / "verifier" / "anchors.json"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert completed.returncode == 0
    assert payload["score"] == 0.0
    assert payload["normalized_score"] == 0.0
    assert payload["protocol_complete"] is False
    assert payload["evaluation"] is None
    assert "342 completed" in payload["disqualification_reason"]


def test_full_hidden_matrix_contract_has_342_evaluation_trajectories(
    tmp_path: Path,
) -> None:
    config = tmp_path / "full_deterministic.toml"
    config.write_text(
        f"""
version = "1.0"

[evaluation]
backend = "deterministic"
games = "atari57"
obs_modes = "both"
seeds = [10001, 20001, 30001]
max_steps = 1
hns_table = "{TASK_DIR / "verifier" / "hns_normalization.csv"}"
require_complete_search_ledger = false
""".strip()
        + "\n"
    )
    output = tmp_path / "full.json"
    subprocess.run(
        [
            sys.executable,
            str(HIDDEN_EVALUATOR),
            str(STARTER_ARTIFACT),
            "--config",
            str(config),
            "--anchors",
            str(TASK_DIR / "verifier" / "anchors.json"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert payload["evaluation"]["games_evaluated"] == 57
    assert payload["evaluation"]["evaluation_trajectories"] == 342


def test_private_complete_ledger_runs_only_after_protocol_preflight(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    (artifact_root / "manifest.json").write_text(
        (STARTER_ARTIFACT / "manifest.json").read_text()
    )
    (artifact_root / "policy.py").write_text(
        (STARTER_ARTIFACT / "policy.py").read_text()
    )
    records = []
    for index, trajectory in enumerate(expected_search_trajectories()):
        relative = Path("searches") / str(index)
        evidence = artifact_root / relative
        evidence.mkdir(parents=True)
        for filename in ("policy.py", "trials.jsonl", "summary.csv", "README.md"):
            (evidence / filename).write_text("fixture\n")
        (evidence / "sample_efficiency.png").write_bytes(b"fixture")
        records.append(
            {
                "env_id": trajectory.env_id,
                "obs_mode": trajectory.obs_mode,
                "repeat_index": trajectory.repeat_index,
                "cumulative_env_steps": FRAME_BUDGET_PER_SEARCH,
                "cumulative_episodes": 1,
                "status": "complete",
                "evidence_path": str(relative),
            }
        )
    (artifact_root / "interaction_ledger.json").write_text(
        json.dumps({"schema_version": "1.0", "records": records})
    )

    config = tmp_path / "private.toml"
    config.write_text(
        f"""
version = "1.0"

[evaluation]
backend = "deterministic"
games = ["Breakout-v5"]
obs_modes = ["native_obs"]
seeds = [101, 202, 303]
max_steps = 4
hns_table = "{TASK_DIR / "verifier" / "hns_normalization.csv"}"
require_complete_search_ledger = true
""".strip()
        + "\n"
    )
    output = tmp_path / "private.json"
    subprocess.run(
        [
            sys.executable,
            str(HIDDEN_EVALUATOR),
            str(artifact_root),
            "--config",
            str(config),
            "--anchors",
            str(TASK_DIR / "verifier" / "anchors.json"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert payload["protocol_complete"] is True
    assert payload["evaluation"] is not None
    assert payload["interaction_budget"]["completed_trajectories"] == 342
    assert payload["interaction_budget"]["counted_env_steps"] == 6_840_000_000


def test_native_task_package_and_public_validation_entrypoint(
    tmp_path: Path,
) -> None:
    assert (
        validate_task(
            TASK_DIR,
            runtime_source=RUNTIME_SOURCE,
        )
        == []
    )

    output = tmp_path / "validation.json"
    subprocess.run(
        [
            sys.executable,
            str(PUBLIC_EVALUATOR),
            "--artifact",
            str(STARTER_ARTIFACT),
            "--validate-only",
            "--json-output-file",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text())
    assert payload["valid"] is True
    assert payload["policy_slots"] == 342
    assert payload["interaction_budget"]["planned_trajectories"] == 342

    task_text = (TASK_DIR / "task.md").read_text()
    full_config = tomllib.loads((TASK_DIR / "verifier" / "config.toml").read_text())
    assert full_config["evaluation"]["require_complete_search_ledger"] is True
    assert "57 games x 2 observation modes x 3 repeats = 342" in task_text
    assert "6,840,000,000" in task_text
    assert "0.8283015254994576" in task_text
    assert "1.1813031161473089" in task_text
