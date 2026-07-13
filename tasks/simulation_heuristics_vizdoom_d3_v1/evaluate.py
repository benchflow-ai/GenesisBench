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

from genesisbench.vizdoom import evaluate_vizdoom_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a screen-CV VizDoom D3 battle policy."
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("final_policy/policy.py"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1050)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--json-output-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_vizdoom_policy(
        args.policy,
        scenario="d3",
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        frame_skip=args.frame_skip,
        render_width=args.render_width,
        render_height=args.render_height,
        failure_return=0.0,
    )
    rendered = result.to_json()
    print(rendered)
    if args.json_output_file is not None:
        args.json_output_file.parent.mkdir(parents=True, exist_ok=True)
        args.json_output_file.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

