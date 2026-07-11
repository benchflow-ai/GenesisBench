#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TODO: independent final evaluator."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.toml",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.artifact.is_file():
        raise FileNotFoundError(args.artifact)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    raise NotImplementedError(
        "Implement clean final evaluation and write machine-readable metrics."
    )


if __name__ == "__main__":
    main()

