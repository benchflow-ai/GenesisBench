#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS_ROOT = REPO_ROOT / "tasks"
DEFAULT_RUNTIME_SOURCE = REPO_ROOT / "src" / "genesisbench"


def _safe_relative_path(value: str, *, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path: {value!r}")
    return path


def prepare_task(
    task_name: str,
    output: Path,
    *,
    force: bool = False,
    tasks_root: Path = DEFAULT_TASKS_ROOT,
    runtime_source: Path = DEFAULT_RUNTIME_SOURCE,
) -> Path:
    source = (tasks_root / task_name).resolve()
    if not source.is_dir() or task_name.startswith("_"):
        raise FileNotFoundError(source)

    config = tomllib.loads((source / "task.toml").read_text())
    starter_path = _safe_relative_path(
        config["starter"]["path"],
        field="starter.path",
    )
    submission_directory = _safe_relative_path(
        config["submission"]["directory"],
        field="submission.directory",
    )

    output = output.resolve()
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists; pass --force")
        shutil.rmtree(output)

    shutil.copytree(
        source,
        output,
        ignore=shutil.ignore_patterns(
            "verifier",
            "__pycache__",
            "*.pyc",
        ),
    )
    runtime_destination = output / "_runtime" / "genesisbench"
    shutil.copytree(
        runtime_source,
        runtime_destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    starter = output / starter_path
    submission = output / submission_directory
    if starter.is_dir():
        shutil.copytree(starter, submission)
    elif starter.is_file():
        submission.mkdir(parents=True)
        shutil.copy2(starter, submission / starter.name)
    else:
        raise FileNotFoundError(starter)

    if (output / "verifier").exists():
        raise RuntimeError("Prepared workspace unexpectedly contains verifier/")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an isolated public GenesisBench task workspace."
    )
    parser.add_argument("task_name")
    parser.add_argument("output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=DEFAULT_TASKS_ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--runtime-source",
        type=Path,
        default=DEFAULT_RUNTIME_SOURCE,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = prepare_task(
        args.task_name,
        args.output,
        force=args.force,
        tasks_root=args.tasks_root.resolve(),
        runtime_source=args.runtime_source.resolve(),
    )
    print(output)


if __name__ == "__main__":
    main()

