#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the GenesisBench Simulation Heuristics Ant v1 leaderboard."
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "simulation_heuristics_ant_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "simulation_heuristics_ant_v1_leaderboard.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.leaderboard.read_text())["rows"]
    rows = sorted(rows, key=lambda row: row["score"])

    names = [row["model"] for row in rows]
    scores = [row["score"] for row in rows]
    normalized = [row["normalized_score"] for row in rows]
    colors = [
        "#a8adb5",
        "#858c96",
        "#646c78",
        "#111827",
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": "#374151",
            "xtick.color": "#6b7280",
            "ytick.color": "#111827",
        }
    )
    figure, axis = plt.subplots(figsize=(12, 7.2), dpi=160)
    figure.patch.set_facecolor("#faf9f6")
    axis.set_facecolor("#faf9f6")

    bars = axis.barh(names, scores, color=colors, height=0.62)
    axis.axvline(
        2382.2277700037166,
        color="#9ca3af",
        linewidth=1.3,
        linestyle=(0, (4, 4)),
        label="Starter anchor",
    )
    axis.axvline(
        3071.5626964052326,
        color="#4b5563",
        linewidth=1.3,
        linestyle=(0, (4, 4)),
        label="Reference anchor",
    )

    for bar, score, normalized_score in zip(
        bars,
        scores,
        normalized,
        strict=True,
    ):
        axis.text(
            score + 28,
            bar.get_y() + bar.get_height() / 2,
            f"{score:,.0f}  ({normalized_score:+.1f})",
            va="center",
            ha="left",
            fontsize=11,
            color="#111827",
            fontweight="bold",
        )

    axis.set_xlim(0, max(scores) * 1.18)
    axis.set_xlabel("Final hidden-suite return")
    axis.set_title(
        "GenesisBench Simulation Heuristics Ant v1 — Final Leaderboard",
        loc="left",
        fontsize=20,
        fontweight="bold",
        color="#111827",
        pad=20,
    )
    axis.text(
        0,
        1.025,
        "OpenHands · 30 minutes/model · highest reasoning · "
        "70% nominal + 30% dynamics robustness",
        transform=axis.transAxes,
        fontsize=10.5,
        color="#6b7280",
    )
    axis.text(
        1,
        -0.13,
        "Labels: raw score (normalized score)",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#6b7280",
        ha="right",
    )

    axis.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=11)
    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=9.5,
        labelcolor="#4b5563",
    )

    figure.subplots_adjust(left=0.20, right=0.96, top=0.82, bottom=0.16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(
        args.output.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
