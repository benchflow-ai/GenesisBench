#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchflow._utils.task_authoring import task_digest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = REPO_ROOT / "experiments" / "article_suite" / "models.toml"
TASKS = (
    "simulation_heuristics_ant_v1",
    "simulation_heuristics_pong_ram_v1",
    "simulation_heuristics_breakout_ram_v1",
    "simulation_heuristics_breakout_rgb_v1",
    "simulation_heuristics_halfcheetah_v1",
    "simulation_heuristics_vizdoom_d1_v1",
    "simulation_heuristics_vizdoom_d3_v1",
    "simulation_heuristics_atari57_v1",
    "simulation_heuristics_montezuma_v1",
)
TASK_LABELS = {
    "simulation_heuristics_ant_v1": "Ant",
    "simulation_heuristics_pong_ram_v1": "Pong",
    "simulation_heuristics_breakout_ram_v1": "Breakout RAM",
    "simulation_heuristics_breakout_rgb_v1": "Breakout RGB",
    "simulation_heuristics_halfcheetah_v1": "HalfCheetah",
    "simulation_heuristics_vizdoom_d1_v1": "VizDoom D1",
    "simulation_heuristics_vizdoom_d3_v1": "VizDoom D3",
    "simulation_heuristics_atari57_v1": "Atari57",
    "simulation_heuristics_montezuma_v1": "Montezuma's Revenge",
}
TASK_RAW_METRICS = {
    "simulation_heuristics_ant_v1": {
        "label": "Weighted hidden return",
        "unit": "return",
    },
    "simulation_heuristics_pong_ram_v1": {
        "label": "Native Pong score",
        "unit": "points",
    },
    "simulation_heuristics_breakout_ram_v1": {
        "label": "Native Breakout return",
        "unit": "points",
    },
    "simulation_heuristics_breakout_rgb_v1": {
        "label": "Native Breakout return",
        "unit": "points",
    },
    "simulation_heuristics_halfcheetah_v1": {
        "label": "Weighted hidden return",
        "unit": "return",
    },
    "simulation_heuristics_vizdoom_d1_v1": {
        "label": "Native D1 mean reward",
        "unit": "reward",
    },
    "simulation_heuristics_vizdoom_d3_v1": {
        "label": "Native D3 mean reward",
        "unit": "reward",
    },
    "simulation_heuristics_atari57_v1": {
        "label": "Median best-mode HNS",
        "unit": "HNS",
    },
    "simulation_heuristics_montezuma_v1": {
        "label": "Capped native return",
        "unit": "points",
    },
}
PROVIDER_LABELS = {
    "azure": "Azure direct",
    "claude_oauth": "Claude OAuth via pinned OpenCode plugin",
}
FINAL_LEADERBOARD_ID = "final"
IQM_TRIM_FRACTION = 0.25
FINAL_DISPLAY_OFFSET = 100.0
TASK_DIGEST_COMPATIBILITY = {
    "simulation_heuristics_ant_v1": {
        "sha256:bbb533da0cb86459f4d49dee667e6c73ac54c0188bc40e54e911d50ef3c3bc38": (
            "Score-equivalent to the current task. Later changes add a "
            "fail-closed verifier timeout plus a CI-only smoke config and "
            "documentation; the publication scoring config is unchanged."
        ),
        "sha256:9da0e00147cf66804e6c2fc17869606bea8f260850c5447989a8880eef940d45": (
            "Score-equivalent to the current task. The later change adds only "
            "a CI smoke config and documentation; the publication scoring "
            "config is unchanged."
        ),
    },
    "simulation_heuristics_halfcheetah_v1": {
        "sha256:80c439f53e4ab964f9d7443cd7fb8f25cf6645a0bc288b0496b871c1800ebe78": (
            "Score-equivalent to the current task. The later loader fix "
            "registers submitted modules before execution so postponed "
            "dataclass annotations import correctly; scoring is unchanged."
        )
    },
    "simulation_heuristics_vizdoom_d1_v1": {
        "sha256:b3431b238bec4e66af6189d30bcd15d5cd227144dfe8bf0dd1011aa5416c1436": (
            "Score-equivalent to the current task. The later loader fix "
            "registers submitted modules before execution so postponed "
            "dataclass annotations import correctly; scoring is unchanged."
        )
    },
    "simulation_heuristics_vizdoom_d3_v1": {
        "sha256:fcc6ed05673b7981aa17d60ca4e0d355fbcb57a6cbca293c0fa201ab97cdd081": (
            "Score-equivalent to the current task. The later loader fix "
            "registers submitted modules before execution so postponed "
            "dataclass annotations import correctly; scoring is unchanged."
        )
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build nine task leaderboards and their final IQM score."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "runs" / "article_suite",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "article_suite.json",
    )
    return parser.parse_args()


def _result_files(model_root: Path) -> list[Path]:
    jobs = model_root / "jobs"
    candidates = sorted(
        path
        for path in jobs.glob("*/results.jsonl")
        if path.parent.name != jobs.name
    )
    if not candidates:
        candidates = sorted(jobs.glob("results.jsonl"))
    return candidates


def _load_results(
    model_root: Path,
    *,
    expected_tasks: tuple[str, ...] = TASKS,
    allow_partial_errors: bool = False,
) -> dict[str, dict[str, Any]]:
    files = _result_files(model_root)
    if len(files) != 1:
        raise RuntimeError(
            f"{model_root} must contain exactly one top-level results.jsonl; "
            f"found {len(files)}"
        )
    rows: dict[str, dict[str, Any]] = {}
    for line in files[0].read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_name = row.get("info", {}).get("task_name")
        if task_name not in expected_tasks:
            continue
        error = row.get("error")
        rollout_dir = row.get("info", {}).get("rollout_dir")
        score_details = (
            Path(rollout_dir) / "verifier" / "genesis-score.json"
            if isinstance(rollout_dir, str)
            else None
        )
        has_score_details = bool(
            score_details is not None and score_details.is_file()
        )
        if error is not None and not (
            isinstance(error, dict)
            and error.get("error") == "missing_llm_trajectory"
        ) and not has_score_details:
            if allow_partial_errors:
                continue
            raise RuntimeError(f"{model_root.name}/{task_name} contains an error")
        reward = row.get("reward")
        if (
            not isinstance(reward, int | float)
            or isinstance(reward, bool)
            or not math.isfinite(float(reward))
        ):
            raise RuntimeError(
                f"{model_root.name}/{task_name} has invalid reward {reward!r}"
            )
        rows[task_name] = row
    missing = sorted(set(expected_tasks) - set(rows))
    if missing and not allow_partial_errors:
        raise RuntimeError(
            f"{model_root.name} is missing article-suite tasks: "
            + ", ".join(missing)
        )
    return rows


def _normalized_task_score(row: dict[str, Any]) -> float:
    rollout_dir = row.get("info", {}).get("rollout_dir")
    if isinstance(rollout_dir, str):
        details_path = Path(rollout_dir) / "verifier" / "genesis-score.json"
        if details_path.is_file():
            details = json.loads(details_path.read_text())
            normalized = details.get("normalized_score")
            if (
                isinstance(normalized, int | float)
                and not isinstance(normalized, bool)
                and math.isfinite(float(normalized))
            ):
                return float(normalized)
    return 100.0 * float(row["reward"])


def _sanitize_score_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "submitted_artifact"
                if key in {"policy_path", "artifact_path"}
                else _sanitize_score_paths(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_score_paths(item) for item in value]
    return value


def _digest_compatibility_note(
    task: str,
    recorded_digest: str,
    current_digest: str,
) -> str | None:
    if recorded_digest == current_digest:
        return None
    return TASK_DIGEST_COMPATIBILITY.get(task, {}).get(recorded_digest)


def _expected_models() -> dict[str, dict[str, Any]]:
    payload = tomllib.loads(MODELS_PATH.read_text())
    return {model["id"]: model for model in payload["models"]}


def _rank_rows(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(row[score_key]),
            str(row["model_id"]),
        ),
    )
    previous_score: float | None = None
    current_rank = 0
    for position, row in enumerate(ranked, start=1):
        score = float(row[score_key])
        if previous_score is None or score != previous_score:
            current_rank = position
        row["rank"] = current_rank
        previous_score = score
    return ranked


def _interquartile_mean(scores: list[float]) -> float:
    if not scores:
        raise ValueError("IQM requires at least one score")
    ordered = sorted(float(score) for score in scores)
    trim_count = int(len(ordered) * IQM_TRIM_FRACTION)
    retained = ordered[trim_count : len(ordered) - trim_count]
    if not retained:
        raise ValueError("IQM trimming removed every score")
    return math.fsum(retained) / len(retained)


def _aggregate_task_scores(task_scores: dict[str, float]) -> dict[str, float]:
    scores = [float(task_scores[task]) for task in TASKS]
    return {
        "final_normalized_score": _interquartile_mean(scores),
        "arithmetic_mean_normalized_score": math.fsum(scores) / len(scores),
        "median_normalized_score": float(statistics.median(scores)),
    }


def _build_leaderboards(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    leaderboards: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = [
            {
                "model_id": row["model_id"],
                "model": row["model"],
                "model_route": row["model_route"],
                "provider": row["provider"],
                "provider_label": row["provider_label"],
                "harness": row["harness"],
                "provider_reasoning_effort": row[
                    "provider_reasoning_effort"
                ],
                "normalized_score": row["task_scores"][task],
                "raw_score": row["raw_task_scores"][task],
                "starter_score": row["task_anchors"][task][
                    "starter_score"
                ],
                "reference_score": row["task_anchors"][task][
                    "reference_score"
                ],
                "submission_detail": row["submission_details"][task],
                "source_run": row["source_runs"][task],
            }
            for row in rows
        ]
        leaderboards.append(
            {
                "id": task,
                "label": TASK_LABELS[task],
                "metric": "raw_score",
                "raw_metric": TASK_RAW_METRICS[task],
                "rows": _rank_rows(
                    task_rows,
                    score_key="raw_score",
                ),
            }
        )

    final_rows = [
        {
            "model_id": row["model_id"],
            "model": row["model"],
            "model_route": row["model_route"],
            "provider": row["provider"],
            "provider_label": row["provider_label"],
            "harness": row["harness"],
            "provider_reasoning_effort": row[
                "provider_reasoning_effort"
            ],
            "final_normalized_score": row["final_normalized_score"],
            "arithmetic_mean_normalized_score": row[
                "arithmetic_mean_normalized_score"
            ],
            "median_normalized_score": row["median_normalized_score"],
            "positive_display_score": row["final_normalized_score"]
            + FINAL_DISPLAY_OFFSET,
            "average_normalized_score": row["average_normalized_score"],
        }
        for row in rows
    ]
    leaderboards.append(
        {
            "id": FINAL_LEADERBOARD_ID,
            "label": "Final normalized score",
            "metric": "final_normalized_score",
            "display_metric": "positive_display_score",
            "display_transform": {
                "type": "additive_offset",
                "offset": FINAL_DISPLAY_OFFSET,
                "formula": "positive_display_score = final_normalized_score + 100",
                "purpose": "plot_only",
            },
            "rows": _rank_rows(
                final_rows,
                score_key="final_normalized_score",
            ),
        }
    )
    return leaderboards


def _leaderboard_relative_path(path: str) -> str:
    detail_path = Path(path)
    try:
        detail_path = detail_path.relative_to("leaderboard")
    except ValueError:
        pass
    return detail_path.as_posix()


def _render_article_suite_markdown(
    leaderboards: list[dict[str, Any]],
) -> str:
    final_board = leaderboards[-1]
    if final_board["id"] != FINAL_LEADERBOARD_ID:
        raise ValueError("Final leaderboard must be last")
    return "\n".join(
        [
        "# GenesisBench Learning Beyond Gradients Article Suite",
        "",
        "The first image contains the nine independently ranked task "
        "leaderboards. The second image contains the final cross-task ranking.",
        "",
        "The nine task panels use each environment's native raw score. "
        "The final scientific metric remains unbounded IQM.",
        "",
        "## Nine task leaderboards",
        "",
        "![Nine task-specific GenesisBench leaderboards]"
        "(article_suite_task_leaderboards.png)",
        "",
        "## Final normalized score",
        "",
        "The primary score is the interquartile mean (IQM): sort the nine task "
        "scores, remove the lowest two and highest two, then average the middle "
        "five. The image uses a plot-only positive display index equal to "
        "`IQM + 100`; raw IQM, arithmetic mean, and median remain in the JSON.",
        "",
        "![Final GenesisBench article-suite leaderboard]"
        "(article_suite_final_leaderboard.png)",
        "",
        "Machine-readable rankings and score-detail paths are available in "
        "[`article_suite.json`](article_suite.json). The scoring rationale is "
        "documented in "
        "[`docs/article-suite-scoring.md`](../docs/article-suite-scoring.md).",
        "",
        ]
    )


def _latest_model_runs(runs_root: Path) -> dict[str, Path]:
    selected: dict[str, tuple[float, Path]] = {}
    for metadata_path in runs_root.glob("*/*/run_metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("dry_run"):
            continue
        model_id = metadata.get("model", {}).get("id")
        finished_at = metadata.get("finished_at")
        if not isinstance(model_id, str) or not isinstance(
            finished_at,
            int | float,
        ):
            continue
        current = selected.get(model_id)
        if current is None or float(finished_at) > current[0]:
            selected[model_id] = (float(finished_at), metadata_path.parent)
    return {model_id: item[1] for model_id, item in selected.items()}


def _latest_task_results(
    runs_root: Path,
) -> dict[
    str,
    dict[str, tuple[float, dict[str, Any], Path, str]],
]:
    selected: dict[
        str,
        dict[str, tuple[float, dict[str, Any], Path, str]],
    ] = {}
    for metadata_path in runs_root.glob("*/*/run_metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("dry_run"):
            continue
        model_id = metadata.get("model", {}).get("id")
        finished_at = metadata.get("finished_at")
        configured_tasks = metadata.get("tasks")
        if (
            not isinstance(model_id, str)
            or not isinstance(finished_at, int | float)
            or not isinstance(configured_tasks, list)
            or not configured_tasks
            or not all(task in TASKS for task in configured_tasks)
        ):
            continue
        model_root = metadata_path.parent
        manifest_path = model_root / "task_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        manifest_digests = {
            item.get("task_id"): item.get("digest")
            for item in manifest.get("tasks", [])
            if isinstance(item, dict)
        }
        if any(
            not isinstance(manifest_digests.get(task), str)
            for task in configured_tasks
        ):
            continue
        try:
            rows = _load_results(
                model_root,
                expected_tasks=tuple(configured_tasks),
                allow_partial_errors=True,
            )
        except RuntimeError:
            continue
        model_rows = selected.setdefault(model_id, {})
        for task, row in rows.items():
            current = model_rows.get(task)
            if current is None or float(finished_at) > current[0]:
                model_rows[task] = (
                    float(finished_at),
                    row,
                    model_root,
                    manifest_digests[task],
                )
    return selected


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    expected_models = _expected_models()
    task_results = _latest_task_results(runs_root)
    missing_models = sorted(set(expected_models) - set(task_results))
    if missing_models:
        raise RuntimeError(
            "Missing completed article-suite model runs: "
            + ", ".join(missing_models)
        )
    ranked: list[dict[str, Any]] = []
    submissions_root = args.output.parent / "article_suite_submissions"
    for model_id in expected_models:
        model_results = task_results[model_id]
        missing_tasks = sorted(set(TASKS) - set(model_results))
        if missing_tasks:
            raise RuntimeError(
                f"{model_id} is missing completed task results: "
                + ", ".join(missing_tasks)
            )
        stale_tasks = [
            task
            for task in TASKS
            if _digest_compatibility_note(
                task,
                model_results[task][3],
                task_digest(REPO_ROOT / "tasks" / task),
            )
            is None
            and model_results[task][3]
            != task_digest(REPO_ROOT / "tasks" / task)
        ]
        if stale_tasks:
            raise RuntimeError(
                f"{model_id} has stale task digests: "
                + ", ".join(stale_tasks)
            )
        latest_task = max(
            model_results.values(),
            key=lambda item: item[0],
        )
        metadata = json.loads(
            (latest_task[2] / "run_metadata.json").read_text()
        )
        task_scores = {
            task: _normalized_task_score(model_results[task][1])
            for task in TASKS
        }
        raw_task_scores: dict[str, float] = {}
        task_anchors: dict[str, dict[str, float]] = {}
        submission_details: dict[str, str] = {}
        for task in TASKS:
            _, result_row, model_root, digest = model_results[task]
            current_digest = task_digest(REPO_ROOT / "tasks" / task)
            compatibility_note = _digest_compatibility_note(
                task,
                digest,
                current_digest,
            )
            task_metadata = json.loads(
                (model_root / "run_metadata.json").read_text()
            )
            rollout_dir = Path(result_row["info"]["rollout_dir"])
            source_score = rollout_dir / "verifier" / "genesis-score.json"
            if not source_score.is_file():
                raise RuntimeError(
                    f"{model_id}/{task} has no verifier genesis-score.json"
                )
            destination = submissions_root / model_id / task
            destination.mkdir(parents=True, exist_ok=True)
            sanitized_score = _sanitize_score_paths(
                json.loads(source_score.read_text())
            )
            raw_score = sanitized_score.get("score")
            starter_score = sanitized_score.get("starter_score")
            reference_score = sanitized_score.get("reference_score")
            for name, value in (
                ("score", raw_score),
                ("starter_score", starter_score),
                ("reference_score", reference_score),
            ):
                if (
                    not isinstance(value, int | float)
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise RuntimeError(
                        f"{model_id}/{task} has invalid {name} {value!r}"
                    )
            raw_task_scores[task] = float(raw_score)
            task_anchors[task] = {
                "starter_score": float(starter_score),
                "reference_score": float(reference_score),
            }
            (destination / "score.json").write_text(
                json.dumps(sanitized_score, indent=2, sort_keys=True) + "\n"
            )
            source_run = str(model_root.relative_to(REPO_ROOT))
            metadata_payload = {
                "benchmark": "learning_beyond_gradients_article_suite",
                "task": task,
                "task_digest": digest,
                "current_task_digest": current_digest,
                "digest_compatibility_note": compatibility_note,
                "model": task_metadata["model"],
                "harness": task_metadata["harness"],
                "provider_reasoning_effort": task_metadata[
                    "provider_reasoning_effort"
                ],
                "normalized_score": task_scores[task],
                "source_run_id": source_run,
                "finished_at": task_metadata["finished_at"],
                "tool_calls": result_row.get("metrics", {}).get(
                    "n_tool_calls",
                    result_row.get("total_tool_calls"),
                ),
                "token_usage": result_row.get("token_usage"),
            }
            (destination / "metadata.json").write_text(
                json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n"
            )
            submission_details[task] = str(
                (destination / "score.json").relative_to(REPO_ROOT)
            )
        aggregate_scores = _aggregate_task_scores(task_scores)
        ranked.append(
            {
                "model_id": metadata["model"]["id"],
                "model": metadata["model"]["display_name"],
                "model_route": metadata["model"]["model"],
                "provider": metadata["model"]["provider"],
                "provider_label": PROVIDER_LABELS.get(
                    metadata["model"]["provider"],
                    metadata["model"]["provider"],
                ),
                "harness": metadata["harness"],
                "provider_reasoning_effort": metadata[
                    "provider_reasoning_effort"
                ],
                **aggregate_scores,
                # Backward-compatible alias for consumers of the first
                # published schema. This is not the primary ranking metric.
                "average_normalized_score": aggregate_scores[
                    "arithmetic_mean_normalized_score"
                ],
                "task_scores": task_scores,
                "raw_task_scores": raw_task_scores,
                "task_anchors": task_anchors,
                "submission_details": submission_details,
                "source_runs": {
                    task: str(
                        model_results[task][2].relative_to(REPO_ROOT)
                    )
                    for task in TASKS
                },
            }
        )
    ranked = _rank_rows(
        ranked,
        score_key="final_normalized_score",
    )
    leaderboards = _build_leaderboards(ranked)

    payload = {
        "benchmark": "learning_beyond_gradients_article_suite",
        "task_count": len(TASKS),
        "leaderboard_count": len(leaderboards),
        "tasks": list(TASKS),
        "aggregation": {
            "primary_metric": "interquartile_mean",
            "primary_field": "final_normalized_score",
            "trim_fraction_per_tail": IQM_TRIM_FRACTION,
            "trimmed_score_count_per_tail": int(
                len(TASKS) * IQM_TRIM_FRACTION
            ),
            "retained_score_count": len(TASKS)
            - 2 * int(len(TASKS) * IQM_TRIM_FRACTION),
            "score_bounds": "unbounded",
            "secondary_fields": [
                "arithmetic_mean_normalized_score",
                "median_normalized_score",
            ],
            "uncertainty": "not_estimated_single_run_per_model_task",
            "display_transform": {
                "type": "additive_offset",
                "offset": FINAL_DISPLAY_OFFSET,
                "formula": "positive_display_score = final_normalized_score + 100",
                "purpose": "plot_only",
                "ranking_field": "final_normalized_score",
            },
        },
        "leaderboards": leaderboards,
        "inference_settings": {
            "field": "provider_reasoning_effort",
            "interpretation": "provider_specific_categorical_setting",
            "cross_provider_comparability": (
                "labels_are_not_a_shared_numeric_compute_scale"
            ),
            "models": [
                {
                    "model_id": row["model_id"],
                    "model": row["model"],
                    "model_route": row["model_route"],
                    "provider": row["provider"],
                    "provider_label": row["provider_label"],
                    "harness": row["harness"],
                    "provider_reasoning_effort": row[
                        "provider_reasoning_effort"
                    ],
                }
                for row in ranked
            ],
        },
        "task_digest_compatibility": TASK_DIGEST_COMPATIBILITY,
        "source_runs": {
            model_id: {
                task: str(task_results[model_id][task][2].relative_to(REPO_ROOT))
                for task in TASKS
            }
            for model_id in expected_models
        },
        "generated_at": datetime.fromtimestamp(
            max(
                item[0]
                for model_results in task_results.values()
                for item in model_results.values()
            ),
            UTC,
        ).isoformat(),
        "rows": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    website_data = REPO_ROOT / "website" / "assets" / "article_suite.json"
    website_data.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output, website_data)

    (args.output.parent / "ARTICLE_SUITE.md").write_text(
        _render_article_suite_markdown(leaderboards)
    )
    try:
        from scripts.plot_article_suite_leaderboards import (
            render_article_suite_leaderboards,
        )
    except ModuleNotFoundError:
        from plot_article_suite_leaderboards import (
            render_article_suite_leaderboards,
        )

    render_article_suite_leaderboards(
        payload,
        leaderboard_dir=args.output.parent,
    )
    print(args.output)


if __name__ == "__main__":
    main()
