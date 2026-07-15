#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_IMAGE_NAME = "article_suite_task_leaderboards.png"
FINAL_IMAGE_NAME = "article_suite_final_leaderboard.png"
MODEL_COLORS = {
    "gpt-5.5": "#111111",
    "gpt-5.6-sol": "#444444",
    "claude-opus-4.8": "#777777",
    "gpt-5.4-mini": "#b5b5b5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot article-suite task and final leaderboards."
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "article_suite.json",
    )
    parser.add_argument(
        "--leaderboard-dir",
        type=Path,
        default=REPO_ROOT / "leaderboard",
    )
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#d1d1d1",
            "axes.labelcolor": "#333333",
            "xtick.color": "#666666",
            "ytick.color": "#171717",
            "text.color": "#171717",
        }
    )


def _score_limits(task_boards: list[dict[str, Any]]) -> tuple[float, float]:
    scores = [
        float(row["raw_score"])
        for board in task_boards
        for row in board["rows"]
    ]
    lower = min(-25.0, 25.0 * math.floor((min(scores) - 10.0) / 25.0))
    upper = max(125.0, 25.0 * math.ceil((max(scores) + 10.0) / 25.0))
    return lower, upper


def _plot_task_leaderboards(
    task_boards: list[dict[str, Any]],
    output: Path,
) -> None:
    _style()
    figure, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=160)
    figure.patch.set_facecolor("#fafafa")

    for axis, board in zip(axes.flat, task_boards, strict=True):
        axis.set_facecolor("#fafafa")
        rows = board["rows"]
        names = [row["model"] for row in rows]
        scores = [float(row["raw_score"]) for row in rows]
        score_stddevs = [
            float(row.get("raw_score_stddev", 0.0)) for row in rows
        ]
        starter_score = float(rows[0]["starter_score"])
        reference_score = float(rows[0]["reference_score"])
        all_values = [
            *(score - stddev for score, stddev in zip(scores, score_stddevs)),
            *(score + stddev for score, stddev in zip(scores, score_stddevs)),
            starter_score,
            reference_score,
        ]
        value_min = min(all_values)
        value_max = max(all_values)
        padding = max(
            (value_max - value_min) * 0.35,
            max(abs(value_min), abs(value_max)) * 0.08,
            1.0,
        )
        lower = min(0.0, value_min - padding)
        upper = max(0.0, value_max + padding)
        span = upper - lower
        colors = [
            MODEL_COLORS.get(row["model_id"], "#666666") for row in rows
        ]
        bars = axis.barh(
            names,
            scores,
            xerr=score_stddevs,
            color=colors,
            height=0.62,
            error_kw={"ecolor": "#333333", "elinewidth": 0.8, "capsize": 2},
        )
        axis.invert_yaxis()
        axis.axvline(
            starter_score,
            color="#777777",
            linewidth=1.0,
            linestyle=(0, (1, 3)),
        )
        axis.axvline(
            reference_score,
            color="#8a8a8a",
            linewidth=1.0,
            linestyle=(0, (4, 4)),
        )
        axis.set_xlim(lower, upper)
        axis.set_title(
            board["label"],
            loc="left",
            fontsize=15,
            fontweight="bold",
            pad=10,
        )
        axis.text(
            0,
            1.01,
            board["raw_metric"]["label"],
            transform=axis.transAxes,
            fontsize=8,
            color="#777777",
        )
        axis.grid(axis="x", color="#e3e3e3", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, labelsize=9.5)
        axis.tick_params(axis="x", labelsize=8.5)
        for bar, score, score_stddev in zip(
            bars,
            scores,
            score_stddevs,
            strict=True,
        ):
            if score >= 0:
                x = score + score_stddev + span * 0.012
                alignment = "left"
            else:
                x = score - score_stddev - span * 0.012
                alignment = "right"
            axis.text(
                x,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.1f} ± {score_stddev:.1f}",
                va="center",
                ha=alignment,
                fontsize=8.5,
                fontweight="bold",
                color="#171717",
            )

    figure.suptitle(
        "GenesisBench — Nine Task Leaderboards",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=24,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.955,
        "Native raw score mean ± sample SD · task-specific axes · "
        "raw-null fail-closed trials use starter-equivalent anchors",
        fontsize=11,
        color="#666666",
    )
    legend_handles = [
        Patch(facecolor=color, label=model_id)
        for model_id, color in MODEL_COLORS.items()
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.text(
        0.945,
        0.02,
        "Dotted: starter · dashed: article reference",
        ha="right",
        fontsize=9,
        color="#777777",
    )
    figure.tight_layout(rect=(0.035, 0.055, 0.98, 0.935))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
        metadata={"Software": "GenesisBench"},
    )
    plt.close(figure)


def _plot_final_leaderboard(
    final_board: dict[str, Any],
    output: Path,
) -> None:
    _style()
    rows = final_board["rows"]
    names = [f"{row['rank']}  {row['model']}" for row in rows]
    scores = [float(row["positive_display_score"]) for row in rows]
    score_stddevs = [
        float(row.get("final_normalized_score_stddev", 0.0)) for row in rows
    ]
    colors = [
        MODEL_COLORS.get(row["model_id"], "#666666") for row in rows
    ]

    figure, axis = plt.subplots(figsize=(12, 7.2), dpi=160)
    figure.patch.set_facecolor("#fafafa")
    axis.set_facecolor("#fafafa")
    bars = axis.barh(
        names,
        scores,
        xerr=score_stddevs,
        color=colors,
        height=0.62,
        error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capsize": 3},
    )
    axis.invert_yaxis()
    lower = 0.0
    upper = max(
        160.0,
        math.ceil(
            (
                max(
                    score + stddev
                    for score, stddev in zip(scores, score_stddevs)
                )
                + 10.0
            )
            / 10.0
        )
        * 10.0,
    )
    span = upper - lower
    axis.set_xlim(lower, upper)
    axis.axvline(
        100,
        color="#777777",
        linewidth=1.0,
        linestyle=(0, (1, 3)),
    )
    axis.grid(axis="x", color="#e3e3e3", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=12)
    axis.tick_params(axis="x", labelsize=10)
    axis.set_xlabel("Positive display index (IQM + 100)", fontsize=11)

    for bar, row, score, score_stddev in zip(
        bars,
        rows,
        scores,
        score_stddevs,
        strict=True,
    ):
        label_color = (
            "#171717" if row["model_id"] == "gpt-5.4-mini" else "#ffffff"
        )
        axis.text(
            score - span * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f} ± {score_stddev:.2f}  "
            f"(IQM {row['final_normalized_score']:.2f})",
            va="center",
            ha="right",
            fontsize=10.5,
            fontweight="bold",
            color=label_color,
        )

    axis.set_title(
        "GenesisBench — Final Article-Suite Leaderboard",
        loc="left",
        fontsize=22,
        fontweight="bold",
        pad=22,
    )
    axis.text(
        0,
        1.025,
        "Plot-only positive index = IQM + 100 · ranking and gaps unchanged · "
        "dotted line = starter-level aggregate",
        transform=axis.transAxes,
        fontsize=10.5,
        color="#666666",
    )
    figure.subplots_adjust(left=0.25, right=0.97, top=0.82, bottom=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
        metadata={"Software": "GenesisBench"},
    )
    plt.close(figure)


def render_article_suite_leaderboards(
    payload: dict[str, Any],
    *,
    leaderboard_dir: Path,
) -> tuple[Path, Path]:
    boards = payload["leaderboards"]
    task_boards = boards[:-1]
    final_board = boards[-1]
    if len(task_boards) != 9 or final_board["id"] != "final":
        raise ValueError("Expected nine task boards followed by final board")

    task_output = leaderboard_dir / TASK_IMAGE_NAME
    final_output = leaderboard_dir / FINAL_IMAGE_NAME
    _plot_task_leaderboards(task_boards, task_output)
    _plot_final_leaderboard(final_board, final_output)
    return task_output, final_output


def main() -> None:
    args = parse_args()
    payload = json.loads(args.leaderboard.read_text())
    outputs = render_article_suite_leaderboards(
        payload,
        leaderboard_dir=args.leaderboard_dir,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
