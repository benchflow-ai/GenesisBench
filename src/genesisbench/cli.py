from __future__ import annotations

import argparse
from pathlib import Path

from genesisbench.ant import DynamicsVariant, evaluate_ant_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Ant policy.")
    parser.add_argument("policy", type=Path)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds or list(range(args.episodes))
    result = evaluate_ant_policy(
        args.policy,
        seeds=seeds,
        max_steps=args.max_steps,
        variants=(DynamicsVariant(),),
    )
    rendered = result.to_json()
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

