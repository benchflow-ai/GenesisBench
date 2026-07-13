#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from genesisbench.montezuma import (  # noqa: E402
    MontezumaEvaluation,
    MontezumaVariant,
    evaluate_montezuma_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hidden Montezuma boundary evaluation."
    )
    parser.add_argument("policy", type=Path)
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
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _relative_policy(
    anchors_path: Path,
    anchor: dict,
    *,
    name: str,
) -> Path:
    relative_path = anchor.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{name} must declare a policy path")
    path = anchors_path.parent / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _configured_variants(evaluation: dict) -> dict[str, MontezumaVariant]:
    variants: dict[str, MontezumaVariant] = {}
    for configured in evaluation["variants"]:
        variant = MontezumaVariant(
            name=str(configured["name"]),
            bootstrap_steps=int(configured.get("bootstrap_steps", 0)),
            pre_policy_noops=int(configured.get("pre_policy_noops", 0)),
        )
        if variant.name in variants:
            raise ValueError(f"Duplicate variant name: {variant.name}")
        variants[variant.name] = variant
    return variants


def _evaluate_raw(
    policy: Path,
    *,
    evaluation: dict,
    reference_policy: Path,
) -> tuple[dict[str, tuple[float, MontezumaEvaluation]], float]:
    variants = _configured_variants(evaluation)
    suites: dict[str, tuple[float, MontezumaEvaluation]] = {}
    weighted_score = 0.0
    total_weight = 0.0

    for suite in evaluation["suites"]:
        suite_name = str(suite["name"])
        weight = float(suite["weight"])
        selected_variants = tuple(variants[str(name)] for name in suite["variants"])
        result = evaluate_montezuma_policy(
            policy,
            seeds=suite["seeds"],
            max_steps=int(evaluation["max_steps"]),
            variants=selected_variants,
            bootstrap_policy_path=reference_policy,
            target_score=float(evaluation["target_score"]),
            failure_score=float(evaluation["failure_score"]),
        )
        suites[suite_name] = (weight, result)
        weighted_score += weight * result.capped_mean_score
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("Evaluation suite weights must sum to a positive value")
    return suites, weighted_score / total_weight


def _anchor_score(
    anchors: dict,
    name: str,
    *,
    anchors_path: Path,
    evaluation: dict,
    reference_policy: Path,
) -> float:
    anchor = anchors[name]
    score = anchor.get("score")
    if isinstance(score, int | float):
        return float(score)
    policy = _relative_policy(anchors_path, anchor, name=name)
    _, calibrated_score = _evaluate_raw(
        policy,
        evaluation=evaluation,
        reference_policy=reference_policy,
    )
    return calibrated_score


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text())
    anchors = json.loads(args.anchors.read_text())
    evaluation = config["evaluation"]
    reference_policy = _relative_policy(
        args.anchors,
        anchors["reference_policy"],
        name="reference_policy",
    )

    suites, score = _evaluate_raw(
        args.policy,
        evaluation=evaluation,
        reference_policy=reference_policy,
    )
    starter_score = _anchor_score(
        anchors,
        "starter_policy",
        anchors_path=args.anchors,
        evaluation=evaluation,
        reference_policy=reference_policy,
    )
    reference_score = _anchor_score(
        anchors,
        "reference_policy",
        anchors_path=args.anchors,
        evaluation=evaluation,
        reference_policy=reference_policy,
    )
    if reference_score == starter_score:
        raise ValueError("starter and reference anchors must have different scores")
    normalized_score = round(
        100.0 * (score - starter_score) / (reference_score - starter_score),
        9,
    )
    suite_payload = {
        name: {
            "weight": weight,
            "score": result.capped_mean_score,
            "evaluation": result.to_dict(),
        }
        for name, (weight, result) in suites.items()
    }
    payload = {
        "score": score,
        "normalized_score": normalized_score,
        "starter_score": starter_score,
        "reference_score": reference_score,
        "target_score": float(evaluation["target_score"]),
        "suites": suite_payload,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
