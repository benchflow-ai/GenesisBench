from __future__ import annotations

import shutil
from pathlib import Path

from scripts.create_task import create_task
from scripts.prepare_task import prepare_task
from scripts.validate_tasks import validate_task


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "tasks"
RUNTIME_SOURCE = REPO_ROOT / "src" / "genesisbench"


def test_reference_task_validates() -> None:
    assert validate_task(
        TASKS_ROOT / "ant_v1",
        runtime_source=RUNTIME_SOURCE,
    ) == []


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
    assert "warehouse_pick_v1" in (task / "task.toml").read_text()
    assert validate_task(task, runtime_source=RUNTIME_SOURCE) == []

    prepared = prepare_task(
        "warehouse_pick_v1",
        tmp_path / "prepared",
        tasks_root=tasks_root,
        runtime_source=RUNTIME_SOURCE,
    )
    assert not (prepared / "verifier").exists()
    assert (prepared / "final_artifact" / "artifact.py").is_file()
    assert (prepared / "_runtime" / "genesisbench").is_dir()


def test_prepare_reference_task_excludes_verifier(tmp_path: Path) -> None:
    prepared = prepare_task(
        "ant_v1",
        tmp_path / "ant",
        tasks_root=TASKS_ROOT,
        runtime_source=RUNTIME_SOURCE,
    )

    assert not (prepared / "verifier").exists()
    assert (
        prepared / "final_policy" / "policy.py"
    ).read_bytes() == (
        TASKS_ROOT / "ant_v1" / "starter_policy" / "policy.py"
    ).read_bytes()

