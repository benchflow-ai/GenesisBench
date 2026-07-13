#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
RUNTIME_CANDIDATES = (
    TASK_DIR / "_runtime",
    TASK_DIR.parent / "src",
    TASK_DIR.parent.parent / "src",
)
for candidate in RUNTIME_CANDIDATES:
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break

from genesisbench.breakout import evaluate_breakout_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a pixel-only Breakout policy."
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("final_policy/policy.py"),
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_breakout_policy(
        args.policy,
        observation_mode="rgb",
        seeds=range(args.seed, args.seed + args.episodes),
        max_steps=args.max_steps,
    )
    rendered = result.to_json()
    print(rendered)
    if args.json_output_file is not None:
        args.json_output_file.parent.mkdir(parents=True, exist_ok=True)
        args.json_output_file.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
