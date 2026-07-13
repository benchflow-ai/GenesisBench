#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
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
        description="Run hidden VizDoom D3 evaluation."
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


def _evaluate_suite_isolated(
    policy: Path,
    *,
    evaluation: dict[str, Any],
    suite: dict[str, Any],
    workspace_root: Path,
    run_label: str,
    suite_index: int,
) -> dict[str, Any]:
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
    return json.loads(output_path.read_text())


def _evaluate_raw(
    policy: Path,
    *,
    evaluation: dict[str, Any],
    workspace_root: Path,
    run_label: str,
) -> tuple[dict[str, dict[str, Any]], float]:
    results: dict[str, dict[str, Any]] = {}
    score = 0.0
    for suite_index, suite in enumerate(evaluation["suites"]):
        result = _evaluate_suite_isolated(
            policy,
            evaluation=evaluation,
            suite=suite,
            workspace_root=workspace_root,
            run_label=run_label,
            suite_index=suite_index,
        )
        results[suite["name"]] = result
        score += float(suite["weight"]) * float(result["mean_return"])
    return results, score


def _validate_numeric_anchor_calibration(
    anchors: dict[str, Any],
    *,
    evaluation: dict[str, Any],
) -> None:
    numeric_anchors = [
        anchor
        for name in ("starter_policy", "reference_policy")
        if isinstance((anchor := anchors[name]).get("score"), int | float)
    ]
    if not numeric_anchors:
        return
    calibration = anchors.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("numeric anchors require calibration metadata")
    if (
        calibration.get("envpool_version")
        != VIZDOOM_ARTICLE_ENVPOOL_VERSION
    ):
        raise ValueError("numeric anchors use the wrong EnvPool version")
    if calibration.get("evaluation") != evaluation:
        raise ValueError(
            "numeric anchors do not match the selected hidden evaluation"
        )


def _anchor_score(
    anchors: dict[str, Any],
    name: str,
    *,
    anchors_path: Path,
    evaluation: dict[str, Any],
    workspace_root: Path,
) -> float:
    anchor = anchors[name]
    score = anchor.get("score")
    if isinstance(score, int | float):
        suite_means = anchor.get("suite_mean_returns")
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
    _, calibrated_score = _evaluate_raw(
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
    _validate_numeric_anchor_calibration(anchors, evaluation=evaluation)
    with tempfile.TemporaryDirectory(
        prefix="genesisbench-vizdoom-d3-hidden-"
    ) as temporary_directory:
        workspace_root = Path(temporary_directory)
        suites, score = _evaluate_raw(
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
        )
        reference_score = _anchor_score(
            anchors,
            "reference_policy",
            anchors_path=args.anchors,
            evaluation=evaluation,
            workspace_root=workspace_root,
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
        "suites": suites,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
