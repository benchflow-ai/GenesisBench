#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS_ROOT = REPO_ROOT / "tasks"
TASK_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _replace_placeholders(
    root: Path,
    *,
    task_name: str,
    title: str,
    author: str,
) -> None:
    replacements = {
        "{{task_name}}": task_name,
        "{{task_title}}": title,
        "{{author}}": author,
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            continue
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        path.write_text(content)


def create_task(
    task_name: str,
    *,
    title: str,
    author: str,
    tasks_root: Path = DEFAULT_TASKS_ROOT,
) -> Path:
    if not TASK_NAME_PATTERN.fullmatch(task_name):
        raise ValueError(
            "Task name must match ^[a-z][a-z0-9_]*$ "
            "(for example: warehouse_pick_v1)."
        )
    template = tasks_root / "_template"
    if not template.is_dir():
        raise FileNotFoundError(template)
    destination = tasks_root / task_name
    if destination.exists():
        raise FileExistsError(destination)

    shutil.copytree(
        template,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    _replace_placeholders(
        destination,
        task_name=task_name,
        title=title,
        author=author,
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a GenesisBench task from tasks/_template."
    )
    parser.add_argument("task_name")
    parser.add_argument("--title", required=True)
    parser.add_argument("--author", default="Your Name")
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=DEFAULT_TASKS_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = create_task(
        args.task_name,
        title=args.title,
        author=args.author,
        tasks_root=args.tasks_root.resolve(),
    )
    print(destination)
    print("Next: edit the TODOs, then run scripts/validate_tasks.py.")


if __name__ == "__main__":
    main()

