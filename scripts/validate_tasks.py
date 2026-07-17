#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from benchflow.task import TaskDocument as BenchFlowTaskDocument
from genesisbench.task_document import TaskDocument

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
    "task.md",
    "evaluate.py",
)
REQUIRED_METADATA = (
    "category",
    "difficulty",
    "tags",
    "reference_task",
    "genesisbench",
)
FORBIDDEN_SPLIT_FILES = (
    "task.toml",
    "instruction.md",
    "prompt.md",
)
FORBIDDEN_AGENT_CONTEXT_FRAGMENTS = (
    "github.com/",
    "raw.githubusercontent.com/",
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
        benchflow_document = BenchFlowTaskDocument.from_path(
            task_directory / "task.md"
        )
        document = TaskDocument.from_path(task_directory / "task.md")
    except Exception as error:
        return [f"invalid task.md: {type(error).__name__}: {error}"]

    if benchflow_document.config.schema_version != "1.3":
        issues.append("schema_version must be '1.3'")
    expected_package_name = f"genesisbench/{task_name}"
    if document.package_name != expected_package_name:
        issues.append(
            "task.name must match directory: "
            f"{expected_package_name!r}"
        )
    if not document.instruction:
        issues.append("task.md prompt body must not be empty")

    metadata = document.frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        issues.append("missing metadata mapping")
        metadata = {}
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

    genesisbench = metadata.get("genesisbench", {})
    if not isinstance(genesisbench, dict):
        issues.append("metadata.genesisbench must be a mapping")
        genesisbench = {}

    try:
        starter_path = _safe_relative_path(
            genesisbench["starter"]["path"],
            field="metadata.genesisbench.starter.path",
        )
    except (KeyError, TypeError, ValueError) as error:
        issues.append(str(error))
        starter_path = None

    try:
        submission_directory = _safe_relative_path(
            genesisbench["submission"]["directory"],
            field="metadata.genesisbench.submission.directory",
        )
        submission_entrypoint = _safe_relative_path(
            genesisbench["submission"]["entrypoint"],
            field="metadata.genesisbench.submission.entrypoint",
        )
    except (KeyError, TypeError, ValueError) as error:
        issues.append(str(error))
        submission_directory = None
        submission_entrypoint = None

    if benchflow_document.config.agent.timeout_sec is None:
        issues.append("agent.timeout_sec is required")
    agent_config = document.frontmatter.get("agent", {})
    verifier_config = document.frontmatter.get("verifier", {})
    environment_config = document.frontmatter.get("environment", {})
    if agent_config.get("user") != "agent":
        issues.append("agent.user must be agent")
    if agent_config.get("network_mode") != "no-network":
        issues.append("agent.network_mode must be no-network")
    if verifier_config.get("user") != "root":
        issues.append("verifier.user must be root")
    if verifier_config.get("network_mode") != "no-network":
        issues.append("verifier.network_mode must be no-network")
    if verifier_config.get("hardening", {}).get("cleanup_conftests") is not True:
        issues.append("verifier.hardening.cleanup_conftests must be true")
    if environment_config.get("network_mode") != "no-network":
        issues.append("environment.network_mode must be no-network")
    if environment_config.get("allow_internet") is not False:
        issues.append("environment.allow_internet must be false")

    if starter_path is not None:
        starter = task_directory / starter_path
        if not starter.exists():
            issues.append(f"starter path does not exist: {starter_path}")
        if submission_entrypoint is not None and starter.is_dir():
            starter_entrypoint = starter / submission_entrypoint
            if not starter_entrypoint.is_file():
                issues.append(
                    "starter artifact must contain submission entrypoint: "
                    f"{starter_path / submission_entrypoint}"
                )

    verifier_metadata = genesisbench.get("verifier", {})
    if not isinstance(verifier_metadata, dict):
        issues.append("metadata.genesisbench.verifier must be a mapping")
        verifier_metadata = {}
    if not isinstance(
        verifier_metadata.get("supports_private_config"),
        bool,
    ):
        issues.append(
            "metadata.genesisbench.verifier.supports_private_config "
            "must be boolean"
        )
    for field in ("reproduction_config", "anchors"):
        if field not in verifier_metadata:
            continue
        try:
            relative = _safe_relative_path(
                verifier_metadata[field],
                field=f"metadata.genesisbench.verifier.{field}",
            )
        except ValueError as error:
            issues.append(str(error))
            continue
        if not (task_directory / relative).is_file():
            issues.append(f"verifier {field} does not exist: {relative}")
        elif field == "anchors":
            anchors = json.loads((task_directory / relative).read_text())
            for anchor_name in ("starter_policy", "reference_policy"):
                anchor = anchors.get(anchor_name, {})
                if "path" not in anchor:
                    continue
                anchor_path = _safe_relative_path(
                    anchor["path"],
                    field=f"anchors.{anchor_name}.path",
                )
                resolved = (task_directory / relative).parent / anchor_path
                if not resolved.is_file():
                    issues.append(
                        f"anchor policy does not exist: {anchor_path}"
                    )

    task_context = task_directory / "task_context"
    if not task_context.is_dir() or not any(task_context.glob("*.md")):
        issues.append("task_context/ must contain at least one Markdown file")
    else:
        for context_path in task_context.glob("*.md"):
            context_text = context_path.read_text().lower()
            for fragment in FORBIDDEN_AGENT_CONTEXT_FRAGMENTS:
                if fragment in context_text:
                    issues.append(
                        "agent-visible task context contains external source "
                        f"locator {fragment!r}: {context_path.name}"
                    )
    dockerfile_path = task_directory / "environment" / "Dockerfile"
    if not dockerfile_path.is_file():
        issues.append("environment/Dockerfile is required")
    else:
        dockerfile = dockerfile_path.read_text()
        for required_fragment in (
            "security/restrict_exec.c",
            'GENESISBENCH_POLICY_ISOLATION="required"',
            "/app/work",
        ):
            if required_fragment not in dockerfile:
                issues.append(
                    "environment/Dockerfile is missing integrity control: "
                    f"{required_fragment}"
                )
    if not (task_directory / "verifier" / "verifier.md").is_file():
        issues.append("verifier/verifier.md is required")
    verifier_script_path = task_directory / "verifier" / "test.sh"
    if not verifier_script_path.is_file():
        issues.append("verifier/test.sh is required")
    else:
        verifier_script = verifier_script_path.read_text()
        if "genesisbench.integrity" not in verifier_script:
            issues.append("verifier/test.sh must run genesisbench.integrity")
        if "GENESISBENCH_POLICY_ISOLATION=required" not in verifier_script:
            issues.append(
                "verifier/test.sh must require policy isolation"
            )
    if not (task_directory / "verifier" / "integrity.json").is_file():
        issues.append("verifier/integrity.json is required")
    for filename in FORBIDDEN_SPLIT_FILES:
        if (task_directory / filename).exists():
            issues.append(
                f"native task must not contain split-layout mirror: {filename}"
            )

    if issues:
        return issues

    with tempfile.TemporaryDirectory(prefix=f"validate-{task_name}-") as temp:
        prepared = prepare_task(
            task_name,
            Path(temp) / task_name,
            tasks_root=task_directory.parent,
            runtime_source=runtime_source,
        )
        for hidden in ("verifier", "oracle", "evidence"):
            if (prepared / hidden).exists():
                issues.append(
                    f"prepared public workspace contains {hidden}/"
                )
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
        and (path / "task.md").is_file()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate GenesisBench and BenchFlow task contracts."
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
