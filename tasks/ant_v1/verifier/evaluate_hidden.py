#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from genesisbench.ant import DynamicsVariant, evaluate_ant_policy  # noqa: E402


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


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text())
    anchors = json.loads(args.anchors.read_text())
    evaluation = config["evaluation"]
    suites = {suite["name"]: suite for suite in evaluation["suites"]}
    nominal = evaluate_ant_policy(
        args.policy,
        seeds=suites["hidden_nominal"]["seeds"],
        max_steps=evaluation["max_steps"],
        variants=(DynamicsVariant(),),
        failure_return=evaluation["failure_return"],
    )
    variants = tuple(
        DynamicsVariant(**variant) for variant in evaluation["variants"]
    )
    robustness = evaluate_ant_policy(
        args.policy,
        seeds=suites["hidden_robustness"]["seeds"],
        max_steps=evaluation["max_steps"],
        variants=variants,
        failure_return=evaluation["failure_return"],
    )
    score = (
        suites["hidden_nominal"]["weight"] * nominal.mean_return
        + suites["hidden_robustness"]["weight"] * robustness.mean_return
    )
    starter_score = anchors["starter_policy"]["score"]
    reference_score = anchors["reference_policy"]["score"]
    normalized_score = 100.0 * (score - starter_score) / (
        reference_score - starter_score
    )
    payload = {
        "score": score,
        "normalized_score": normalized_score,
        "starter_score": starter_score,
        "reference_score": reference_score,
        "hidden_nominal_weight": suites["hidden_nominal"]["weight"],
        "hidden_robustness_weight": suites["hidden_robustness"]["weight"],
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
