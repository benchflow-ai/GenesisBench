#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path


VERIFIER_DIR = Path(__file__).resolve().parent
RUNTIME_CANDIDATES = [Path("/opt/genesisbench")]
RUNTIME_CANDIDATES.extend(ancestor / "src" for ancestor in VERIFIER_DIR.parents)
for candidate in RUNTIME_CANDIDATES:
    if (candidate / "genesisbench").is_dir():
        sys.path.insert(0, str(candidate))
        break

from genesisbench.ant import (  # noqa: E402
    AntEvaluation,
    DynamicsVariant,
    evaluate_ant_policy,
)


EvaluationBundle = tuple[AntEvaluation, AntEvaluation, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hidden Ant evaluation.")
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.toml",
        help="Evaluation suite config; production can inject a private file.",
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        default=Path(__file__).parent / "anchors.json",
        help="Normalization anchors matching the selected evaluation suite.",
    )
    return parser.parse_args()


def _policy_path(path: Path) -> Path:
    resolved = path.resolve()
    return resolved / "policy.py" if resolved.is_dir() else resolved


def _policy_fingerprint(path: Path) -> str:
    return hashlib.sha256(_policy_path(path).read_bytes()).hexdigest()


def _evaluate_raw(
    policy: Path,
    *,
    evaluation: dict,
) -> EvaluationBundle:
    suites = {suite["name"]: suite for suite in evaluation["suites"]}
    nominal = evaluate_ant_policy(
        policy,
        seeds=suites["hidden_nominal"]["seeds"],
        max_steps=evaluation["max_steps"],
        variants=(DynamicsVariant(),),
        failure_return=evaluation["failure_return"],
    )
    variants = tuple(DynamicsVariant(**variant) for variant in evaluation["variants"])
    robustness = evaluate_ant_policy(
        policy,
        seeds=suites["hidden_robustness"]["seeds"],
        max_steps=evaluation["max_steps"],
        variants=variants,
        failure_return=evaluation["failure_return"],
    )
    score = (
        suites["hidden_nominal"]["weight"] * nominal.mean_return
        + suites["hidden_robustness"]["weight"] * robustness.mean_return
    )
    return nominal, robustness, score


def _evaluate_cached(
    policy: Path,
    *,
    evaluation: dict,
    cache: dict[str, EvaluationBundle],
) -> EvaluationBundle:
    fingerprint = _policy_fingerprint(policy)
    if fingerprint not in cache:
        cache[fingerprint] = _evaluate_raw(
            policy,
            evaluation=evaluation,
        )
    return cache[fingerprint]


def _anchor_score(
    anchors: dict,
    name: str,
    *,
    anchors_path: Path,
    evaluation: dict,
    cache: dict[str, EvaluationBundle],
) -> float:
    anchor = anchors[name]
    score = anchor.get("score")
    if isinstance(score, int | float):
        return float(score)
    relative_path = anchor.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{name} must declare score or path")
    policy = anchors_path.parent / relative_path
    _, _, calibrated_score = _evaluate_cached(
        policy,
        evaluation=evaluation,
        cache=cache,
    )
    return calibrated_score


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text())
    anchors = json.loads(args.anchors.read_text())
    evaluation = config["evaluation"]
    suites = {suite["name"]: suite for suite in evaluation["suites"]}
    cache: dict[str, EvaluationBundle] = {}
    nominal, robustness, score = _evaluate_cached(
        args.policy,
        evaluation=evaluation,
        cache=cache,
    )
    starter_score = _anchor_score(
        anchors,
        "starter_policy",
        anchors_path=args.anchors,
        evaluation=evaluation,
        cache=cache,
    )
    reference_score = _anchor_score(
        anchors,
        "reference_policy",
        anchors_path=args.anchors,
        evaluation=evaluation,
        cache=cache,
    )
    if reference_score == starter_score:
        raise ValueError("starter and reference anchors must have different scores")
    # Round away float noise so the anchor policies normalize to exactly 0/100
    # (score and anchor scores travel different code paths).
    normalized_score = round(
        100.0 * (score - starter_score) / (reference_score - starter_score), 9
    )
    payload = {
        "score": score,
        "normalized_score": normalized_score,
        "starter_score": starter_score,
        "reference_score": reference_score,
        "hidden_nominal_weight": suites["hidden_nominal"]["weight"],
        "hidden_robustness_weight": suites["hidden_robustness"]["weight"],
        "unique_policy_evaluations": len(cache),
        "hidden_nominal": nominal.to_dict(),
        "hidden_robustness": robustness.to_dict(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
