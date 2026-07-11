#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import tomllib
from pathlib import Path

if __package__:
    from scripts.prepare_task import (
        DEFAULT_RUNTIME_SOURCE,
        DEFAULT_TASKS_ROOT,
        prepare_task,
    )
else:
    from prepare_task import (  # type: ignore[no-redef]
        DEFAULT_RUNTIME_SOURCE,
        DEFAULT_TASKS_ROOT,
        prepare_task,
    )


REQUIRED_FILES = (
    "README.md",
    "benchmark.txt",
    "task.toml",
    "prompt.md",
    "evaluate.py",
)
REQUIRED_METADATA = (
    "description",
    "author",
    "category",
    "difficulty",
    "tags",
    "reference_task",
)


def _safe_relative_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path: {value!r}")
    return path


def validate_task(
    task_directory: Path,
    *,
    runtime_source: Path = DEFAULT_RUNTIME_SOURCE,
) -> list[str]:
    issues: list[str] = []
    task_name = task_directory.name

    for required in REQUIRED_FILES:
        if not (task_directory / required).is_file():
            issues.append(f"missing required file: {required}")
    if issues:
        return issues

    try:
        config = tomllib.loads((task_directory / "task.toml").read_text())
    except Exception as error:
        return [f"invalid task.toml: {type(error).__name__}: {error}"]

    if config.get("version") != "1.0":
        issues.append("version must be '1.0'")
    if config.get("name") != task_name:
        issues.append(
            f"task.toml name must match directory: {task_name!r}"
        )
    if not isinstance(config.get("title"), str) or not config["title"].strip():
        issues.append("title must be a non-empty string")

    metadata = config.get("metadata")
    if not isinstance(metadata, dict):
        issues.append("missing [metadata] table")
    else:
        for field in REQUIRED_METADATA:
            if field not in metadata:
                issues.append(f"metadata.{field} is required")
        if "tags" in metadata and not (
            isinstance(metadata["tags"], list)
            and all(isinstance(tag, str) and tag for tag in metadata["tags"])
        ):
            issues.append("metadata.tags must be a list of strings")
        if "reference_task" in metadata and not isinstance(
            metadata["reference_task"],
            bool,
        ):
            issues.append("metadata.reference_task must be boolean")

    try:
        starter_path = _safe_relative_path(
            config["starter"]["path"],
            field="starter.path",
        )
    except (KeyError, TypeError, ValueError) as error:
        issues.append(str(error))
        starter_path = None

    try:
        submission_directory = _safe_relative_path(
            config["submission"]["directory"],
            field="submission.directory",
        )
        submission_entrypoint = _safe_relative_path(
            config["submission"]["entrypoint"],
            field="submission.entrypoint",
        )
    except (KeyError, TypeError, ValueError) as error:
        issues.append(str(error))
        submission_directory = None
        submission_entrypoint = None

    try:
        verifier_entrypoint = _safe_relative_path(
            config["verifier"]["entrypoint"],
            field="verifier.entrypoint",
        )
    except (KeyError, TypeError, ValueError) as error:
        issues.append(str(error))
        verifier_entrypoint = None

    budget = config.get("budget")
    if not isinstance(budget, dict) or not isinstance(
        budget.get("wall_clock_minutes"),
        int,
    ):
        issues.append("budget.wall_clock_minutes must be an integer")
    elif budget["wall_clock_minutes"] <= 0:
        issues.append("budget.wall_clock_minutes must be positive")

    if starter_path is not None:
        starter = task_directory / starter_path
        if not starter.exists():
            issues.append(f"starter.path does not exist: {starter_path}")
        if submission_entrypoint is not None and starter.is_dir():
            starter_entrypoint = starter / submission_entrypoint
            if not starter_entrypoint.is_file():
                issues.append(
                    "starter artifact must contain submission.entrypoint: "
                    f"{starter_path / submission_entrypoint}"
                )

    if verifier_entrypoint is not None and not (
        task_directory / verifier_entrypoint
    ).is_file():
        issues.append(
            f"verifier.entrypoint does not exist: {verifier_entrypoint}"
        )

    verifier = config.get("verifier", {})
    if not isinstance(verifier.get("supports_private_config"), bool):
        issues.append("verifier.supports_private_config must be boolean")
    for field in ("reproduction_config", "anchors"):
        if field not in verifier:
            continue
        try:
            relative = _safe_relative_path(
                verifier[field],
                field=f"verifier.{field}",
            )
        except ValueError as error:
            issues.append(str(error))
            continue
        if not (task_directory / relative).is_file():
            issues.append(f"verifier.{field} does not exist: {relative}")

    task_context = task_directory / "task_context"
    if not task_context.is_dir() or not any(task_context.glob("*.md")):
        issues.append("task_context/ must contain at least one Markdown file")

    if issues:
        return issues

    with tempfile.TemporaryDirectory(prefix=f"validate-{task_name}-") as temp:
        prepared = prepare_task(
            task_name,
            Path(temp) / task_name,
            tasks_root=task_directory.parent,
            runtime_source=runtime_source,
        )
        if (prepared / "verifier").exists():
            issues.append("prepared public workspace contains verifier/")
        if not (prepared / "_runtime" / "genesisbench").is_dir():
            issues.append("prepared public workspace is missing _runtime/")
        if (
            submission_directory is not None
            and submission_entrypoint is not None
            and not (
                prepared / submission_directory / submission_entrypoint
            ).is_file()
        ):
            issues.append("prepared workspace is missing submission entrypoint")

    return issues


def discover_tasks(tasks_root: Path) -> list[Path]:
    return sorted(
        path
        for path in tasks_root.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (path / "task.toml").is_file()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate GenesisBench task structure and public boundary."
    )
    parser.add_argument("--task")
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
    tasks_root = args.tasks_root.resolve()
    if args.task:
        tasks = [tasks_root / args.task]
    else:
        tasks = discover_tasks(tasks_root)

    failed = False
    for task in tasks:
        issues = validate_task(
            task,
            runtime_source=args.runtime_source.resolve(),
        )
        if issues:
            failed = True
            print(f"FAIL {task.name}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"OK   {task.name}")
    if not tasks:
        raise SystemExit("No tasks found.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
