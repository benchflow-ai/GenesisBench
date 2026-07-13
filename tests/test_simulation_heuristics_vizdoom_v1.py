from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib

import numpy as np
import pytest

from genesisbench.vizdoom import (
    D1_ALLOWED_VARIABLES,
    D3_ALLOWED_VARIABLES,
    PolicySourceViolation,
    VIZDOOM_ARTICLE_ENVPOOL_VERSION,
    VizDoomEpisode,
    VizDoomEvaluation,
    _require_article_envpool_version,
    _validate_d1_action,
    _validate_d3_action,
    audit_vizdoom_policy,
    evaluate_vizdoom_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
D1_TASK = REPO_ROOT / "tasks" / "simulation_heuristics_vizdoom_d1_v1"
D3_TASK = REPO_ROOT / "tasks" / "simulation_heuristics_vizdoom_d3_v1"
D1_HIDDEN_EVALUATOR = D1_TASK / "verifier" / "evaluate_hidden.py"
D3_HIDDEN_EVALUATOR = D3_TASK / "verifier" / "evaluate_hidden.py"
RUN_SLOW = os.environ.get("GENESISBENCH_RUN_VIZDOOM_SLOW") == "1"


def _runtime_platform_key() -> str:
    machine = platform.machine().lower()
    machine = {
        "amd64": "x86_64",
        "aarch64": "arm64",
    }.get(machine, machine)
    return f"{sys.platform}-{machine}"


def _load_policy(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"test_policy_{path.parent.parent.name}_{path.parent.name}",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Policy()


@pytest.mark.parametrize("task", [D1_TASK, D3_TASK])
def test_vizdoom_task_validates_and_prepares(task: Path, tmp_path: Path) -> None:
    from scripts.prepare_task import prepare_task
    from scripts.validate_tasks import validate_task

    runtime = REPO_ROOT / "src" / "genesisbench"
    assert validate_task(task, runtime_source=runtime) == []

    prepared = prepare_task(
        task.name,
        tmp_path / task.name,
        tasks_root=task.parent,
        runtime_source=runtime,
    )
    assert not (prepared / "verifier").exists()
    assert not (prepared / "oracle").exists()
    assert (prepared / "final_policy" / "policy.py").is_file()
    assert (
        prepared / "final_policy" / "policy.py"
    ).read_bytes() == (
        task / "starter_policy" / "policy.py"
    ).read_bytes()


def test_policy_input_allowlists_match_article_contracts() -> None:
    assert D1_ALLOWED_VARIABLES == ("HEALTH",)
    assert D3_ALLOWED_VARIABLES == (
        "HEALTH",
        "AMMO2",
        "HITCOUNT",
        "DAMAGECOUNT",
        "KILLCOUNT",
    )


def test_declared_runtime_matches_article_source() -> None:
    assert VIZDOOM_ARTICLE_ENVPOOL_VERSION == "1.1.1"
    for task in (D1_TASK, D3_TASK):
        dockerfile = (task / "environment" / "Dockerfile").read_text()
        anchors = json.loads((task / "verifier" / "anchors.json").read_text())
        assert '"envpool==1.1.1"' in dockerfile
        assert "EnvPool 1.1.1" in anchors["environment"]
        assert "1.2.5" not in dockerfile
        assert "1.2.5" not in anchors["environment"]


def test_runtime_guard_rejects_silent_version_drift() -> None:
    class WrongEnvPool:
        __version__ = "1.2.5"

    with pytest.raises(RuntimeError, match="require EnvPool 1.1.1"):
        _require_article_envpool_version(WrongEnvPool)


def test_action_validation_contracts() -> None:
    assert _validate_d1_action(np.asarray(5)) == 5
    with pytest.raises(ValueError):
        _validate_d1_action(np.asarray([1]))
    with pytest.raises(ValueError):
        _validate_d1_action(6)

    valid = _validate_d3_action(np.asarray([1, 1, 1, 0, 0, 1, 0, -8]))
    assert valid.shape == (8,)
    with pytest.raises(ValueError):
        _validate_d3_action(np.zeros(7))
    with pytest.raises(ValueError):
        _validate_d3_action(np.asarray([2, 0, 0, 0, 0, 0, 0, 0]))


def test_policy_source_audit_rejects_privileged_access(tmp_path: Path) -> None:
    safe = tmp_path / "safe" / "policy.py"
    safe.parent.mkdir()
    safe.write_text(
        "class Policy:\n"
        "    def act(self, frame, variables):\n"
        "        return 0\n"
    )
    audit_vizdoom_policy(safe)

    privileged = tmp_path / "privileged" / "policy.py"
    privileged.parent.mkdir()
    privileged.write_text(
        "import envpool\n"
        "class Policy:\n"
        "    def act(self, frame, variables):\n"
        "        return 0\n"
    )
    with pytest.raises(PolicySourceViolation):
        audit_vizdoom_policy(privileged)


@pytest.mark.parametrize(
    "path",
    [
        D1_TASK / "oracle" / "policy.py",
        D1_TASK / "verifier" / "anchor_policies" / "reference_policy.py",
        D3_TASK / "oracle" / "policy.py",
        D3_TASK / "verifier" / "anchor_policies" / "reference_policy.py",
    ],
)
def test_reference_policy_sources_pass_privilege_audit(path: Path) -> None:
    audit_vizdoom_policy(path)


def test_starter_policies_match_action_shapes() -> None:
    d1 = _load_policy(D1_TASK / "starter_policy" / "policy.py")
    d1_frame = np.zeros((180, 240, 3), dtype=np.uint8)
    assert 0 <= d1.act(d1_frame, {"HEALTH": 100.0}) <= 5

    d3 = _load_policy(D3_TASK / "starter_policy" / "policy.py")
    d3_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    d3_action = d3.act(
        d3_frame,
        {
            "HEALTH": 100.0,
            "AMMO2": 20.0,
            "HITCOUNT": 0.0,
            "DAMAGECOUNT": 0.0,
            "KILLCOUNT": 0.0,
        },
    )
    assert d3_action.shape == (8,)
    _validate_d3_action(d3_action)


def test_hidden_configs_use_unseen_seed_batches() -> None:
    for task in (D1_TASK, D3_TASK):
        config = tomllib.loads((task / "verifier" / "config.toml").read_text())
        seeds = [suite["seed"] for suite in config["evaluation"]["suites"]]
        assert seeds
        assert 0 not in seeds
        assert len(seeds) == len(set(seeds))


def test_anchor_metadata_records_article_targets() -> None:
    d1 = json.loads((D1_TASK / "verifier" / "anchors.json").read_text())
    assert d1["public_article_target"]["envpool_version"] == "1.1.1"
    assert d1["public_article_target"]["mean_return"] == pytest.approx(
        0.9440999741666019
    )
    assert d1["public_article_target"]["min_return"] == pytest.approx(
        0.28999998047947884
    )

    d3 = json.loads((D3_TASK / "verifier" / "anchors.json").read_text())
    assert d3["public_article_target"]["envpool_version"] == "1.1.1"
    assert d3["public_article_target"]["mean_return"] == 557.0
    assert d3["public_article_target"]["min_return"] == 440.0
    assert d3["public_article_target"]["returns"] == [
        545,
        475,
        480,
        440,
        690,
        500,
        600,
        595,
        530,
        715,
    ]


def test_d3_numeric_anchor_calibration_matches_hidden_config() -> None:
    anchors = json.loads(
        (D3_TASK / "verifier" / "anchors.json").read_text()
    )
    config = tomllib.loads(
        (D3_TASK / "verifier" / "config.toml").read_text()
    )
    assert anchors["calibration"] == {
        "envpool_version": "1.1.1",
        "evaluation": config["evaluation"],
    }
    assert anchors["starter_policy"]["score"] == pytest.approx(31.9)
    assert anchors["starter_policy"]["suite_mean_returns"] == {
        "hidden_nominal": 28.0,
        "hidden_secondary": 41.0,
    }
    assert anchors["reference_policy"]["score"] == pytest.approx(331.1)
    assert anchors["reference_policy"]["suite_mean_returns"] == {
        "hidden_nominal": 311.0,
        "hidden_secondary": 378.0,
    }


def test_d1_numeric_anchor_calibration_matches_hidden_config() -> None:
    anchors = json.loads(
        (D1_TASK / "verifier" / "anchors.json").read_text()
    )
    config = tomllib.loads(
        (D1_TASK / "verifier" / "config.toml").read_text()
    )
    assert set(anchors["calibrations"]) == {
        "darwin-arm64",
        "linux-x86_64",
    }
    for calibration in anchors["calibrations"].values():
        assert calibration["envpool_version"] == "1.1.1"
        assert calibration["evaluation"] == config["evaluation"]
        assert calibration["starter_policy"]["score"] == pytest.approx(
            0.44161998964846133
        )
        assert calibration["starter_policy"]["suite_mean_returns"] == {
            "hidden_nominal": pytest.approx(0.5395999886095524),
            "hidden_secondary": pytest.approx(0.21299999207258224),
        }
    darwin = anchors["calibrations"]["darwin-arm64"]
    assert darwin["reference_policy"]["score"] == pytest.approx(
        0.6982399759441613
    )
    assert darwin["reference_policy"]["suite_mean_returns"] == {
        "hidden_nominal": pytest.approx(0.8401999741792678),
        "hidden_secondary": pytest.approx(0.3669999800622463),
    }
    linux = anchors["calibrations"]["linux-x86_64"]
    assert linux["reference_policy"]["score"] == pytest.approx(
        0.8178199748694897
    )
    assert linux["reference_policy"]["suite_mean_returns"] == {
        "hidden_nominal": pytest.approx(0.9847999721765518),
        "hidden_secondary": pytest.approx(0.42819998115301133),
    }
    assert linux["starter_policy"]["score"] == pytest.approx(
        0.44161998964846133
    )


def test_evaluation_json_uses_public_return_key() -> None:
    evaluation = VizDoomEvaluation(
        scenario="d1",
        policy_path="/tmp/policy.py",
        envpool_version="1.1.1",
        batch_seed=0,
        max_steps=1,
        frame_skip=1,
        render_width=240,
        render_height=180,
        episodes=(
            VizDoomEpisode(
                seed=0,
                lane=0,
                return_=1.25,
                length=1,
                terminated=False,
                truncated=True,
                invalid_action=False,
                policy_error=None,
                mean_action_latency_ms=0.1,
                final_variables={"HEALTH": 100.0},
            ),
        ),
    )
    episode = evaluation.to_dict()["episodes"][0]
    assert episode["return"] == 1.25
    assert "return_" not in episode


@pytest.mark.skipif(
    not RUN_SLOW,
    reason="set GENESISBENCH_RUN_VIZDOOM_SLOW=1 for EnvPool regression",
)
def test_d1_article_reference_regression() -> None:
    envpool = pytest.importorskip("envpool")
    pytest.importorskip("cv2")
    assert envpool.__version__ == VIZDOOM_ARTICLE_ENVPOOL_VERSION
    result = evaluate_vizdoom_policy(
        D1_TASK / "oracle" / "policy.py",
        scenario="d1",
        seed=0,
        episodes=10,
        max_steps=2100,
        frame_skip=1,
        render_width=240,
        render_height=180,
        failure_return=-1.0,
    )
    assert result.envpool_version == "1.1.1"
    assert result.mean_return == pytest.approx(0.9440999741666019)
    assert result.min_return == pytest.approx(0.28999998047947884)


@pytest.mark.skipif(
    not RUN_SLOW,
    reason="set GENESISBENCH_RUN_VIZDOOM_SLOW=1 for EnvPool regression",
)
def test_d1_hidden_evaluator_isolates_each_suite_process(
    tmp_path: Path,
) -> None:
    envpool = pytest.importorskip("envpool")
    assert envpool.__version__ == VIZDOOM_ARTICLE_ENVPOOL_VERSION
    evaluation = {
        "scenario": "d1",
        "max_steps": 1,
        "frame_skip": 1,
        "render_width": 240,
        "render_height": 180,
        "failure_return": -1.0,
        "suites": [
            {
                "name": "smoke_a",
                "weight": 0.5,
                "seed": 911,
                "episodes": 1,
            },
            {
                "name": "smoke_b",
                "weight": 0.5,
                "seed": 919,
                "episodes": 1,
            },
        ],
    }
    config = tmp_path / "config.toml"
    config.write_text(
        """version = "1.0"

[evaluation]
scenario = "d1"
max_steps = 1
frame_skip = 1
render_width = 240
render_height = 180
failure_return = -1.0

[[evaluation.suites]]
name = "smoke_a"
weight = 0.5
seed = 911
episodes = 1

[[evaluation.suites]]
name = "smoke_b"
weight = 0.5
seed = 919
episodes = 1
"""
    )
    anchors = tmp_path / "anchors.json"
    anchors.write_text(
        json.dumps(
            {
                "calibrations": {
                    _runtime_platform_key(): {
                        "envpool_version": "1.1.1",
                        "evaluation": evaluation,
                        "starter_policy": {
                            "score": 0.0,
                            "suite_mean_returns": {
                                "smoke_a": 0.0,
                                "smoke_b": 0.0,
                            },
                        },
                        "reference_policy": {
                            "score": 1.0,
                            "suite_mean_returns": {
                                "smoke_a": 1.0,
                                "smoke_b": 1.0,
                            },
                        },
                    },
                },
                "starter_policy": {
                    "path": "unused-starter.py",
                },
                "reference_policy": {
                    "path": "unused-reference.py",
                },
            }
        )
    )
    fake_shm = tmp_path / "dev-shm"
    fake_boost = tmp_path / "boost-interprocess"
    fake_shm.mkdir()
    fake_boost.mkdir()
    stale_size = 2_481_040
    stale_resources = [
        fake_shm / "ViZDoomSMstale-a",
        fake_shm / "ViZDoomSMstale-b",
        fake_shm / "sem.ViZDoomSMstale-c",
        fake_boost / "ViZDoomSMstale-d",
        fake_boost / "ViZDoomSMstale-e",
    ]
    for resource in stale_resources:
        with resource.open("wb") as stream:
            stream.truncate(stale_size)
    unrelated = fake_shm / "unrelated-shared-memory"
    unrelated.write_text("keep")
    (tmp_path / "_vizdoom").mkdir()
    output = tmp_path / "hidden.json"
    worker_environment = os.environ.copy()
    worker_environment["GENESISBENCH_VIZDOOM_RESOURCE_ROOTS"] = (
        f"{fake_shm}{os.pathsep}{fake_boost}"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(D1_HIDDEN_EVALUATOR),
            str(D1_TASK / "oracle" / "policy.py"),
            "--config",
            str(config),
            "--anchors",
            str(anchors),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env=worker_environment,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "Failed to create ./_vizdoom/" not in completed.stdout
    payload = json.loads(output.read_text())
    assert payload["isolation"] == {
        "mode": "subprocess_per_suite",
        "suite_process_count": 2,
        "unique_working_directories": True,
    }
    assert payload["anchor_calibration"]["platform"] == _runtime_platform_key()
    assert payload["resource_cleanup"]["removed_resources"] == len(
        stale_resources
    )
    assert payload["resource_cleanup"]["removed_bytes"] == (
        len(stale_resources) * stale_size
    )
    assert all(not resource.exists() for resource in stale_resources)
    assert unrelated.read_text() == "keep"


@pytest.mark.skipif(
    not RUN_SLOW or sys.platform != "linux",
    reason="full hidden-oracle regression runs on the Linux verifier runtime",
)
def test_d1_hidden_evaluator_oracle_regression(tmp_path: Path) -> None:
    envpool = pytest.importorskip("envpool")
    assert envpool.__version__ == VIZDOOM_ARTICLE_ENVPOOL_VERSION
    (tmp_path / "_vizdoom").mkdir()
    output = tmp_path / "hidden.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(D1_HIDDEN_EVALUATOR),
            str(D1_TASK / "oracle" / "policy.py"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    payload = json.loads(output.read_text())
    anchors = json.loads(
        (D1_TASK / "verifier" / "anchors.json").read_text()
    )
    calibration = anchors["calibrations"][_runtime_platform_key()]
    assert payload["normalized_score"] == 100.0
    assert payload["score"] == pytest.approx(
        calibration["reference_policy"]["score"]
    )
    assert payload["starter_score"] == pytest.approx(
        calibration["starter_policy"]["score"]
    )
    assert payload["reference_score"] == pytest.approx(
        calibration["reference_policy"]["score"]
    )
    assert payload["isolation"]["mode"] == "subprocess_per_suite"
    assert payload["anchor_calibration"]["platform"] == _runtime_platform_key()


@pytest.mark.skipif(
    not RUN_SLOW,
    reason="set GENESISBENCH_RUN_VIZDOOM_SLOW=1 for EnvPool regression",
)
def test_d3_article_reference_regression() -> None:
    envpool = pytest.importorskip("envpool")
    pytest.importorskip("cv2")
    assert envpool.__version__ == VIZDOOM_ARTICLE_ENVPOOL_VERSION
    result = evaluate_vizdoom_policy(
        D3_TASK / "oracle" / "policy.py",
        scenario="d3",
        seed=0,
        episodes=10,
        max_steps=1050,
        frame_skip=2,
        render_width=640,
        render_height=480,
        failure_return=0.0,
    )
    assert result.envpool_version == "1.1.1"
    returns = [episode.return_ for episode in result.episodes]
    assert returns == [
        545.0,
        475.0,
        480.0,
        440.0,
        690.0,
        500.0,
        600.0,
        595.0,
        530.0,
        715.0,
    ]
    assert result.mean_return == 557.0
    assert result.min_return == 440.0


@pytest.mark.skipif(
    not RUN_SLOW,
    reason="set GENESISBENCH_RUN_VIZDOOM_SLOW=1 for EnvPool regression",
)
def test_d3_hidden_evaluator_isolates_each_suite_process(
    tmp_path: Path,
) -> None:
    envpool = pytest.importorskip("envpool")
    assert envpool.__version__ == VIZDOOM_ARTICLE_ENVPOOL_VERSION
    evaluation = {
        "scenario": "d3",
        "max_steps": 1,
        "frame_skip": 2,
        "render_width": 640,
        "render_height": 480,
        "failure_return": 0.0,
        "suites": [
            {
                "name": "smoke_a",
                "weight": 0.5,
                "seed": 901,
                "episodes": 1,
            },
            {
                "name": "smoke_b",
                "weight": 0.5,
                "seed": 907,
                "episodes": 1,
            },
        ],
    }
    config = tmp_path / "config.toml"
    config.write_text(
        """version = "1.0"

[evaluation]
scenario = "d3"
max_steps = 1
frame_skip = 2
render_width = 640
render_height = 480
failure_return = 0.0

[[evaluation.suites]]
name = "smoke_a"
weight = 0.5
seed = 901
episodes = 1

[[evaluation.suites]]
name = "smoke_b"
weight = 0.5
seed = 907
episodes = 1
"""
    )
    anchors = tmp_path / "anchors.json"
    anchors.write_text(
        json.dumps(
            {
                "calibration": {
                    "envpool_version": "1.1.1",
                    "evaluation": evaluation,
                },
                "starter_policy": {
                    "score": 0.0,
                    "suite_mean_returns": {
                        "smoke_a": 0.0,
                        "smoke_b": 0.0,
                    },
                },
                "reference_policy": {
                    "score": 1.0,
                    "suite_mean_returns": {
                        "smoke_a": 1.0,
                        "smoke_b": 1.0,
                    },
                },
            }
        )
    )
    (tmp_path / "_vizdoom").mkdir()
    output = tmp_path / "hidden.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(D3_HIDDEN_EVALUATOR),
            str(D3_TASK / "oracle" / "policy.py"),
            "--config",
            str(config),
            "--anchors",
            str(anchors),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "Failed to create ./_vizdoom/" not in completed.stdout
    payload = json.loads(output.read_text())
    assert payload["isolation"] == {
        "mode": "subprocess_per_suite",
        "suite_process_count": 2,
        "unique_working_directories": True,
    }


@pytest.mark.skipif(
    not RUN_SLOW,
    reason="set GENESISBENCH_RUN_VIZDOOM_SLOW=1 for EnvPool regression",
)
def test_d3_hidden_evaluator_oracle_regression(tmp_path: Path) -> None:
    envpool = pytest.importorskip("envpool")
    assert envpool.__version__ == VIZDOOM_ARTICLE_ENVPOOL_VERSION
    (tmp_path / "_vizdoom").mkdir()
    output = tmp_path / "hidden.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(D3_HIDDEN_EVALUATOR),
            str(D3_TASK / "oracle" / "policy.py"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    payload = json.loads(output.read_text())
    assert payload["normalized_score"] == 100.0
    assert payload["score"] == pytest.approx(331.1)
    assert payload["starter_score"] == pytest.approx(31.9)
    assert payload["reference_score"] == pytest.approx(331.1)
    assert payload["isolation"]["mode"] == "subprocess_per_suite"
