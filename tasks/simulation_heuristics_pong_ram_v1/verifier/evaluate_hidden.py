#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from genesisbench.pong import (  # noqa: E402
    PONG_TARGET_SCORE,
    PongEvaluation,
    PongVariant,
    evaluate_pong_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hidden Atari Pong RAM evaluation."
    )
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


def _configured_variants(evaluation: dict) -> dict[str, PongVariant]:
    variants = {}
    for configured in evaluation["variants"]:
        variant = PongVariant(
            name=configured["name"],
            noop_max=int(configured["noop_max"]),
            frame_skip=int(configured["frame_skip"]),
            repeat_action_probability=float(configured["repeat_action_probability"]),
            use_fire_reset=bool(configured["use_fire_reset"]),
        )
        variants[variant.name] = variant
    return variants


def _evaluate_raw(
    policy: Path,
    *,
    evaluation: dict,
) -> tuple[dict[str, PongEvaluation], float]:
    variants = _configured_variants(evaluation)
    suites: dict[str, PongEvaluation] = {}
    weighted_score = 0.0
    total_weight = 0.0

    for suite in evaluation["suites"]:
        weight = float(suite["weight"])
        variant_name = suite["variant"]
        if variant_name not in variants:
            raise ValueError(
                f"Suite {suite['name']!r} references unknown variant {variant_name!r}"
            )
        result = evaluate_pong_policy(
            policy,
            seeds=suite["seeds"],
            max_steps=int(evaluation["max_steps"]),
            variants=(variants[variant_name],),
            failure_score=float(evaluation["failure_score"]),
        )
        suites[suite["name"]] = result
        weighted_score += weight * result.mean_score
        total_weight += weight

    if not math.isclose(total_weight, 1.0):
        raise ValueError(
            f"Evaluation suite weights must sum to 1.0, got {total_weight}"
        )
    return suites, weighted_score


def _anchor_score(
    anchors: dict,
    name: str,
    *,
    anchors_path: Path,
    evaluation: dict,
    evaluated_policy: Path,
    evaluated_score: float,
) -> float:
    anchor = anchors[name]
    score = anchor.get("score")
    if isinstance(score, int | float):
        return float(score)
    relative_path = anchor.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{name} must declare score or path")
    policy = (anchors_path.parent / relative_path).resolve()
    if policy == evaluated_policy:
        return evaluated_score
    _, calibrated_score = _evaluate_raw(
        policy,
        evaluation=evaluation,
    )
    return calibrated_score


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text())
    anchors = json.loads(args.anchors.read_text())
    evaluation = config["evaluation"]
    suites, score = _evaluate_raw(
        args.policy,
        evaluation=evaluation,
    )
    evaluated_policy = args.policy.resolve()
    if evaluated_policy.is_dir():
        evaluated_policy = evaluated_policy / "policy.py"
    starter_score = _anchor_score(
        anchors,
        "starter_policy",
        anchors_path=args.anchors,
        evaluation=evaluation,
        evaluated_policy=evaluated_policy,
        evaluated_score=score,
    )
    reference_score = _anchor_score(
        anchors,
        "reference_policy",
        anchors_path=args.anchors,
        evaluation=evaluation,
        evaluated_policy=evaluated_policy,
        evaluated_score=score,
    )
    if reference_score == starter_score:
        raise ValueError("starter and reference anchors must have different scores")

    normalized_score = round(
        100.0 * (score - starter_score) / (reference_score - starter_score),
        9,
    )
    suite_weights = {
        suite["name"]: float(suite["weight"]) for suite in evaluation["suites"]
    }
    payload = {
        "score": score,
        "normalized_score": normalized_score,
        "starter_score": starter_score,
        "reference_score": reference_score,
        "target_score": PONG_TARGET_SCORE,
        "suite_weights": suite_weights,
        "suites": {name: result.to_dict() for name, result in suites.items()},
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
