#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import pwd
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any


VERIFIER_DIR = Path(__file__).resolve().parent
RUNTIME_CANDIDATES = [Path("/opt/genesisbench")]
RUNTIME_CANDIDATES.extend(
    ancestor / "src" for ancestor in VERIFIER_DIR.parents
)
for candidate in RUNTIME_CANDIDATES:
    if (candidate / "genesisbench").is_dir():
        sys.path.insert(0, str(candidate))
        break

from genesisbench.vizdoom import (  # noqa: E402
    VIZDOOM_ARTICLE_ENVPOOL_VERSION,
    evaluate_vizdoom_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hidden VizDoom D1 evaluation."
    )
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=VERIFIER_DIR / "config.toml",
        help="Evaluation suite config; production can inject a private file.",
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        default=VERIFIER_DIR / "anchors.json",
        help="Normalization anchors matching the selected evaluation suite.",
    )
    parser.add_argument(
        "--worker-request",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _worker_main(
    policy: Path,
    *,
    request_path: Path,
    output_path: Path | None,
) -> None:
    if output_path is None:
        raise ValueError("--output is required in worker mode")
    request = json.loads(request_path.read_text())
    evaluation = request["evaluation"]
    suite = request["suite"]
    result = evaluate_vizdoom_policy(
        policy,
        scenario=evaluation["scenario"],
        seed=suite["seed"],
        episodes=suite["episodes"],
        max_steps=evaluation["max_steps"],
        frame_skip=evaluation["frame_skip"],
        render_width=evaluation["render_width"],
        render_height=evaluation["render_height"],
        failure_return=evaluation["failure_return"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.to_json() + "\n")


def _safe_name(value: str) -> str:
    rendered = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    ).strip("-")
    return rendered or "suite"


def _resource_roots() -> tuple[Path, ...]:
    override = os.environ.get("GENESISBENCH_VIZDOOM_RESOURCE_ROOTS")
    if override:
        return tuple(
            Path(value)
            for value in override.split(os.pathsep)
            if value
        )
    return (Path("/dev/shm"), Path("/tmp/boost_interprocess"))


def _is_vizdoom_resource(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("vizdoom") or lowered.startswith("sem.vizdoom")


def _task_owned_uids() -> set[int]:
    owners = {os.geteuid()}
    try:
        owners.add(pwd.getpwnam("agent").pw_uid)
    except KeyError:
        pass
    return owners


def _task_owned_stale_processes() -> list[int]:
    if os.geteuid() != 0 or not Path("/app").is_dir():
        return []
    stale: list[int] = []
    current_pid = os.getpid()
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        if pid == current_pid:
            continue
        try:
            command = (
                (process_dir / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
            cwd = Path(os.readlink(process_dir / "cwd"))
            comm = (process_dir / "comm").read_text().strip().lower()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not cwd.is_relative_to("/app"):
            continue
        is_vizdoom = comm == "vizdoom" or "/vizdoom/bin/vizdoom" in command
        is_public_evaluator = (
            "python" in command
            and (
                "/app/evaluate.py" in command
                or " evaluate.py " in f" {command} "
            )
        )
        if is_vizdoom or is_public_evaluator:
            stale.append(pid)
    return stale


def _terminate_processes(pids: list[int]) -> int:
    if not pids:
        return 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (PermissionError, ProcessLookupError):
            pass
    deadline = time.monotonic() + 1.0
    remaining = set(pids)
    while remaining and time.monotonic() < deadline:
        remaining = {pid for pid in remaining if Path(f"/proc/{pid}").exists()}
        if remaining:
            time.sleep(0.05)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            pass
    return len(pids)


def _remove_vizdoom_resources() -> tuple[int, int]:
    removed = 0
    removed_bytes = 0
    allowed_uids = _task_owned_uids()
    overridden = "GENESISBENCH_VIZDOOM_RESOURCE_ROOTS" in os.environ
    for root in _resource_roots():
        if not root.is_dir():
            continue
        for resource in root.iterdir():
            if not _is_vizdoom_resource(resource.name):
                continue
            try:
                metadata = resource.lstat()
            except FileNotFoundError:
                continue
            if not overridden and metadata.st_uid not in allowed_uids:
                continue
            try:
                if resource.is_dir() and not resource.is_symlink():
                    resource.rmdir()
                else:
                    resource.unlink()
            except (FileNotFoundError, OSError):
                continue
            removed += 1
            removed_bytes += metadata.st_size
    return removed, removed_bytes


def _cleanup_stale_vizdoom_state() -> dict[str, int]:
    stale_processes = _task_owned_stale_processes()
    terminated = _terminate_processes(stale_processes)
    removed, removed_bytes = _remove_vizdoom_resources()
    return {
        "terminated_processes": terminated,
        "removed_resources": removed,
        "removed_bytes": removed_bytes,
    }


def _evaluate_suite_isolated(
    policy: Path,
    *,
    evaluation: dict[str, Any],
    suite: dict[str, Any],
    workspace_root: Path,
    run_label: str,
    suite_index: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    cleanup = _cleanup_stale_vizdoom_state()
    workspace = (
        workspace_root
        / f"{_safe_name(run_label)}-{suite_index:02d}-{_safe_name(suite['name'])}"
    )
    workspace.mkdir(parents=True, exist_ok=False)
    request_path = workspace / "request.json"
    output_path = workspace / "result.json"
    request_path.write_text(
        json.dumps(
            {
                "evaluation": evaluation,
                "suite": suite,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(policy.resolve()),
            "--worker-request",
            str(request_path),
            "--output",
            str(output_path),
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Isolated VizDoom suite evaluation failed "
            f"for {run_label}/{suite['name']} with rc={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not output_path.is_file():
        raise RuntimeError(
            f"Isolated VizDoom suite did not write {output_path}"
        )
    return json.loads(output_path.read_text()), cleanup


def _evaluate_raw(
    policy: Path,
    *,
    evaluation: dict[str, Any],
    workspace_root: Path,
    run_label: str,
) -> tuple[dict[str, dict[str, Any]], float, list[dict[str, int]]]:
    results: dict[str, dict[str, Any]] = {}
    cleanup_runs: list[dict[str, int]] = []
    score = 0.0
    for suite_index, suite in enumerate(evaluation["suites"]):
        result, cleanup = _evaluate_suite_isolated(
            policy,
            evaluation=evaluation,
            suite=suite,
            workspace_root=workspace_root,
            run_label=run_label,
            suite_index=suite_index,
        )
        cleanup_runs.append(cleanup)
        results[suite["name"]] = result
        score += float(suite["weight"]) * float(result["mean_return"])
    return results, score, cleanup_runs


def _validate_numeric_anchor_calibration(
    anchors: dict[str, Any],
    *,
    evaluation: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    machine = platform.machine().lower()
    machine = {
        "amd64": "x86_64",
        "aarch64": "arm64",
    }.get(machine, machine)
    platform_key = f"{sys.platform}-{machine}"
    calibrations = anchors.get("calibrations")
    if calibrations is None:
        return platform_key, None
    if not isinstance(calibrations, dict):
        raise ValueError("anchor calibrations must be a mapping")
    calibration = calibrations.get(platform_key)
    if not isinstance(calibration, dict):
        raise ValueError(
            f"no numeric anchor calibration for platform {platform_key}"
        )
    if (
        calibration.get("envpool_version")
        != VIZDOOM_ARTICLE_ENVPOOL_VERSION
    ):
        raise ValueError("numeric anchors use the wrong EnvPool version")
    if calibration.get("evaluation") != evaluation:
        raise ValueError(
            "numeric anchors do not match the selected hidden evaluation"
        )
    for name in ("starter_policy", "reference_policy"):
        numeric_anchor = calibration.get(name)
        if not isinstance(numeric_anchor, dict) or not isinstance(
            numeric_anchor.get("score"),
            int | float,
        ):
            raise ValueError(
                f"platform calibration must declare numeric {name}"
            )
    return platform_key, calibration


def _anchor_score(
    anchors: dict[str, Any],
    name: str,
    *,
    anchors_path: Path,
    evaluation: dict[str, Any],
    workspace_root: Path,
    calibration: dict[str, Any] | None,
) -> float:
    anchor = anchors[name]
    numeric_anchor = calibration.get(name) if calibration is not None else None
    score = (
        numeric_anchor.get("score")
        if isinstance(numeric_anchor, dict)
        else None
    )
    if isinstance(score, int | float):
        suite_means = numeric_anchor.get("suite_mean_returns")
        if not isinstance(suite_means, dict):
            raise ValueError(
                f"numeric anchor {name} must declare suite_mean_returns"
            )
        suite_names = {suite["name"] for suite in evaluation["suites"]}
        if set(suite_means) != suite_names:
            raise ValueError(
                f"numeric anchor {name} has mismatched hidden suites"
            )
        derived_score = sum(
            float(suite["weight"]) * float(suite_means[suite["name"]])
            for suite in evaluation["suites"]
        )
        if not math.isclose(
            derived_score,
            float(score),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"numeric anchor {name} score does not match suite means"
            )
        return float(score)
    relative_path = anchor.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{name} must declare score or path")
    policy = anchors_path.parent / relative_path
    _, calibrated_score, _ = _evaluate_raw(
        policy,
        evaluation=evaluation,
        workspace_root=workspace_root,
        run_label=f"anchor-{name}",
    )
    return calibrated_score


def main() -> None:
    args = parse_args()
    if args.worker_request is not None:
        _worker_main(
            args.policy,
            request_path=args.worker_request,
            output_path=args.output,
        )
        return

    config = tomllib.loads(args.config.read_text())
    anchors = json.loads(args.anchors.read_text())
    evaluation = config["evaluation"]
    calibration_key, calibration = _validate_numeric_anchor_calibration(
        anchors,
        evaluation=evaluation,
    )
    with tempfile.TemporaryDirectory(
        prefix="genesisbench-vizdoom-d1-hidden-"
    ) as temporary_directory:
        workspace_root = Path(temporary_directory)
        suites, score, cleanup_runs = _evaluate_raw(
            args.policy,
            evaluation=evaluation,
            workspace_root=workspace_root,
            run_label="candidate",
        )
        starter_score = _anchor_score(
            anchors,
            "starter_policy",
            anchors_path=args.anchors,
            evaluation=evaluation,
            workspace_root=workspace_root,
            calibration=calibration,
        )
        reference_score = _anchor_score(
            anchors,
            "reference_policy",
            anchors_path=args.anchors,
            evaluation=evaluation,
            workspace_root=workspace_root,
            calibration=calibration,
        )
    if reference_score == starter_score:
        raise ValueError("starter and reference anchors must have different scores")
    normalized_score = round(
        100.0 * (score - starter_score) / (reference_score - starter_score),
        9,
    )
    payload = {
        "score": score,
        "normalized_score": normalized_score,
        "starter_score": starter_score,
        "reference_score": reference_score,
        "isolation": {
            "mode": "subprocess_per_suite",
            "suite_process_count": len(evaluation["suites"]),
            "unique_working_directories": True,
        },
        "anchor_calibration": {
            "mode": "numeric" if calibration is not None else "path",
            "platform": calibration_key,
            "envpool_version": VIZDOOM_ARTICLE_ENVPOOL_VERSION,
        },
        "resource_cleanup": {
            "runs": cleanup_runs,
            "terminated_processes": sum(
                run["terminated_processes"] for run in cleanup_runs
            ),
            "removed_resources": sum(
                run["removed_resources"] for run in cleanup_runs
            ),
            "removed_bytes": sum(
                run["removed_bytes"] for run in cleanup_runs
            ),
        },
        "suites": suites,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
