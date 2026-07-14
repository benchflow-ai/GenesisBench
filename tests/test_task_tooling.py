from __future__ import annotations

import shutil
from pathlib import Path

from scripts.create_task import create_task
from scripts.prepare_task import prepare_task
from scripts.run_article_suite import TASKS as ARTICLE_TASKS
from scripts.validate_tasks import discover_tasks, validate_task


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "tasks"
RUNTIME_SOURCE = REPO_ROOT / "src" / "genesisbench"
RETIRED_OPENHANDS_RUNNER_PATHS = (
    "containers/simulation_heuristics_ant_v1/Dockerfile",
    "experiments/simulation_heuristics_ant_v1/README.md",
    "experiments/simulation_heuristics_ant_v1/models.toml",
    "scripts/build_simulation_heuristics_ant_v1_runner_image.sh",
    "scripts/run_openhands_agent.py",
    "scripts/run_simulation_heuristics_ant_v1_experiment.py",
)


def test_article_suite_contains_exactly_nine_valid_tasks() -> None:
    discovered = discover_tasks(TASKS_ROOT)

    assert {task.name for task in discovered} == set(ARTICLE_TASKS)
    assert len(discovered) == 9
    for task in discovered:
        assert validate_task(
            task,
            runtime_source=RUNTIME_SOURCE,
        ) == []


def test_tasks_own_their_containers_and_legacy_runner_is_removed() -> None:
    for task_name in ARTICLE_TASKS:
        assert (TASKS_ROOT / task_name / "environment" / "Dockerfile").is_file()

    for relative_path in RETIRED_OPENHANDS_RUNNER_PATHS:
        assert not (REPO_ROOT / relative_path).exists()


def test_create_and_prepare_task_scaffold(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    shutil.copytree(TASKS_ROOT / "_template", tasks_root / "_template")
    task = create_task(
        "warehouse_pick_v1",
        title="Warehouse Pick Policy Improvement",
        author="Test Contributor",
        tasks_root=tasks_root,
    )

    assert task.name == "warehouse_pick_v1"
    assert "warehouse_pick_v1" in (task / "task.md").read_text()
    assert not (task / "task.toml").exists()
    assert not (task / "instruction.md").exists()
    assert validate_task(task, runtime_source=RUNTIME_SOURCE) == []

    prepared = prepare_task(
        "warehouse_pick_v1",
        tmp_path / "prepared",
        tasks_root=tasks_root,
        runtime_source=RUNTIME_SOURCE,
    )
    assert not (prepared / "verifier").exists()
    assert not (prepared / "oracle").exists()
    assert not (prepared / "evidence").exists()
    assert (prepared / "final_artifact" / "artifact.py").is_file()
    assert (prepared / "task.md").is_file()
    assert (prepared / "_runtime" / "genesisbench").is_dir()


def test_prepare_reference_task_excludes_verifier(tmp_path: Path) -> None:
    prepared = prepare_task(
        "simulation_heuristics_ant_v1",
        tmp_path / "ant",
        tasks_root=TASKS_ROOT,
        runtime_source=RUNTIME_SOURCE,
    )

    assert not (prepared / "verifier").exists()
    assert not (prepared / "oracle").exists()
    assert (
        prepared / "final_policy" / "policy.py"
    ).read_bytes() == (
        TASKS_ROOT / "simulation_heuristics_ant_v1" / "starter_policy" / "policy.py"
    ).read_bytes()
