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
PROTOCOL_PATH = REPO_ROOT / "experiments" / "article_suite" / "protocol.toml"
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


def _protocol() -> dict[str, Any]:
    return tomllib.loads(PROTOCOL_PATH.read_text())


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


def _aggregate_task_scores(
    task_scores: dict[str, float],
    *,
    trial_task_scores: dict[int, dict[str, float]] | None = None,
) -> dict[str, Any]:
    scores = [float(task_scores[task]) for task in TASKS]
    aggregate: dict[str, Any] = {
        "final_normalized_score": _interquartile_mean(scores),
        "arithmetic_mean_normalized_score": math.fsum(scores) / len(scores),
        "median_normalized_score": float(statistics.median(scores)),
    }
    if trial_task_scores:
        trial_final_scores = {
            trial: _interquartile_mean(
                [float(task_scores_for_trial[task]) for task in TASKS]
            )
            for trial, task_scores_for_trial in sorted(
                trial_task_scores.items()
            )
        }
        values = list(trial_final_scores.values())
        aggregate.update(
            {
                "final_normalized_score": statistics.fmean(values),
                "final_normalized_score_stddev": statistics.stdev(values),
                "trial_final_normalized_scores": trial_final_scores,
            }
        )
    else:
        aggregate.update(
            {
                "final_normalized_score_stddev": 0.0,
                "trial_final_normalized_scores": {
                    1: aggregate["final_normalized_score"]
                },
            }
        )
    return aggregate


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
                "normalized_score_stddev": row[
                    "task_score_stddevs"
                ][task],
                "raw_score": row["raw_task_scores"][task],
                "raw_score_stddev": row["raw_task_score_stddevs"][task],
                "trial_count": row["trial_count"],
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
            "final_normalized_score_stddev": row[
                "final_normalized_score_stddev"
            ],
            "trial_count": row["trial_count"],
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


def _published_artifact_path(path: Path, *, output_parent: Path) -> str:
    path = path.resolve()
    output_parent = output_parent.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.relative_to(output_parent))


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
        "The nine task panels report five-trial mean ± sample standard "
        "deviation for each environment's native raw score.",
        "",
        "## Nine task leaderboards",
        "",
        "![Nine task-specific GenesisBench leaderboards]"
        "(article_suite_task_leaderboards.png)",
        "",
        "## Final normalized score",
        "",
        "Each trial computes an interquartile mean (IQM): sort the nine task "
        "scores, remove the lowest two and highest two, then average the middle "
        "five. The final score is mean ± sample standard deviation across five "
        "trial IQMs. The image uses a plot-only positive display index equal "
        "to `IQM + 100`; raw metrics remain in the JSON.",
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


def _trial_batch_results(
    batch_root: Path,
    *,
    expected_models: set[str],
    expected_trials: int,
) -> dict[
    str,
    dict[str, dict[int, tuple[dict[str, Any], dict[str, Any], Path, str]]],
]:
    selected: dict[
        str,
        dict[str, dict[int, tuple[dict[str, Any], dict[str, Any], Path, str]]],
    ] = {}
    metadata_paths = [
        *batch_root.glob("*/trial-*/run_metadata.json"),
        *batch_root.glob("*/trial-*/*/run_metadata.json"),
    ]
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        model_id = metadata.get("model", {}).get("id")
        trial = metadata.get("trial")
        configured_tasks = metadata.get("tasks")
        if (
            model_id not in expected_models
            or not isinstance(trial, int)
            or not 1 <= trial <= expected_trials
            or not isinstance(configured_tasks, list)
            or not configured_tasks
            or not all(task in TASKS for task in configured_tasks)
            or metadata.get("status") != "completed"
            or metadata.get("return_code") != 0
            or metadata.get("dry_run")
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
        try:
            rows = _load_results(
                model_root,
                expected_tasks=tuple(configured_tasks),
            )
        except RuntimeError:
            continue
        for task, row in rows.items():
            digest = manifest_digests.get(task)
            if not isinstance(digest, str):
                continue
            task_trials = selected.setdefault(model_id, {}).setdefault(
                task,
                {},
            )
            current = task_trials.get(trial)
            candidate = (
                metadata,
                row,
                model_root,
                digest,
            )
            if (
                current is None
                or float(metadata["finished_at"])
                > float(current[0]["finished_at"])
            ):
                task_trials[trial] = candidate
    return selected


def _trial_batch_missing(
    results: dict[
        str,
        dict[str, dict[int, tuple[dict[str, Any], dict[str, Any], Path, str]]],
    ],
    *,
    expected_models: set[str],
    expected_trials: int,
) -> list[str]:
    missing = []
    for model_id in sorted(expected_models):
        for task in TASKS:
            trials = results.get(model_id, {}).get(task, {})
            for trial in range(1, expected_trials + 1):
                if trial not in trials:
                    missing.append(f"{model_id}/{task}/trial-{trial:02d}")
    return missing


def _latest_complete_model_batches(
    runs_root: Path,
    *,
    expected_models: set[str],
    protocol: dict[str, Any],
) -> tuple[
    dict[str, Path],
    dict[
        str,
        dict[
            str,
            dict[int, tuple[dict[str, Any], dict[str, Any], Path, str]],
        ],
    ],
]:
    expected_trials = int(protocol["trials"])
    manifests = []
    for manifest_path in runs_root.glob("*/batch_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("protocol") == protocol:
            manifests.append(manifest_path.parent)

    selected_batches: dict[str, Path] = {}
    selected_results: dict[
        str,
        dict[
            str,
            dict[int, tuple[dict[str, Any], dict[str, Any], Path, str]],
        ],
    ] = {}
    for model_id in sorted(expected_models):
        candidates = []
        incomplete: list[tuple[Path, int]] = []
        for batch_root in manifests:
            results = _trial_batch_results(
                batch_root,
                expected_models={model_id},
                expected_trials=expected_trials,
            )
            missing = _trial_batch_missing(
                results,
                expected_models={model_id},
                expected_trials=expected_trials,
            )
            if missing:
                incomplete.append((batch_root, len(missing)))
                continue
            finished_at = max(
                entry[0]["finished_at"]
                for task_results in results[model_id].values()
                for entry in task_results.values()
            )
            candidates.append(
                (float(finished_at), batch_root, results[model_id])
            )
        if not candidates:
            detail = (
                ", ".join(
                    f"{path.name}: {missing_count} missing"
                    for path, missing_count in sorted(incomplete)[-3:]
                )
                or "no matching protocol batches"
            )
            raise RuntimeError(
                f"No complete {expected_trials}-trial article-suite batch is "
                f"available for {model_id}: {detail}"
            )
        _, batch_root, model_results = max(
            candidates,
            key=lambda item: item[0],
        )
        selected_batches[model_id] = batch_root
        selected_results[model_id] = model_results
    return selected_batches, selected_results


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    expected_models = _expected_models()
    protocol = _protocol()
    trial_count = int(protocol["trials"])
    model_batch_roots, task_results = _latest_complete_model_batches(
        runs_root,
        expected_models=set(expected_models),
        protocol=protocol,
    )
    ranked: list[dict[str, Any]] = []
    submissions_root = args.output.parent / "article_suite_submissions"
    for model_id in expected_models:
        model_results = task_results[model_id]
        representative_metadata = model_results[TASKS[0]][1][0]
        task_scores: dict[str, float] = {}
        task_score_stddevs: dict[str, float] = {}
        raw_task_scores: dict[str, float] = {}
        raw_task_score_stddevs: dict[str, float] = {}
        task_anchors: dict[str, dict[str, float]] = {}
        submission_details: dict[str, str] = {}
        source_runs: dict[str, list[str]] = {}
        trial_task_scores = {
            trial: {} for trial in range(1, trial_count + 1)
        }
        for task in TASKS:
            trial_results = model_results[task]
            current_digest = task_digest(REPO_ROOT / "tasks" / task)
            destination = submissions_root / model_id / task
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=True)
            normalized_values = []
            raw_values = []
            starter_values = []
            reference_values = []
            trial_records = []
            for trial in range(1, trial_count + 1):
                task_metadata, result_row, model_root, digest = trial_results[
                    trial
                ]
                compatibility_note = _digest_compatibility_note(
                    task,
                    digest,
                    current_digest,
                )
                if digest != current_digest and compatibility_note is None:
                    raise RuntimeError(
                        f"{model_id}/{task}/trial-{trial:02d} has stale "
                        f"task digest {digest}"
                    )
                rollout_dir = Path(result_row["info"]["rollout_dir"])
                source_score = (
                    rollout_dir / "verifier" / "genesis-score.json"
                )
                if not source_score.is_file():
                    raise RuntimeError(
                        f"{model_id}/{task}/trial-{trial:02d} has no "
                        "verifier genesis-score.json"
                    )
                sanitized_score = _sanitize_score_paths(
                    json.loads(source_score.read_text())
                )
                normalized = _normalized_task_score(result_row)
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
                            f"{model_id}/{task}/trial-{trial:02d} has "
                            f"invalid {name} {value!r}"
                        )
                trial_destination = destination / f"trial-{trial:02d}"
                trial_destination.mkdir(parents=True, exist_ok=True)
                trial_score_path = trial_destination / "score.json"
                trial_score_path.write_text(
                    json.dumps(
                        sanitized_score,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                source_run = str(model_root.relative_to(REPO_ROOT))
                trial_metadata = {
                    "benchmark": "learning_beyond_gradients_article_suite",
                    "task": task,
                    "trial": trial,
                    "task_digest": digest,
                    "current_task_digest": current_digest,
                    "digest_compatibility_note": compatibility_note,
                    "model": task_metadata["model"],
                    "harness": task_metadata["harness"],
                    "provider_reasoning_effort": task_metadata[
                        "provider_reasoning_effort"
                    ],
                    "normalized_score": normalized,
                    "raw_score": float(raw_score),
                    "source_run_id": source_run,
                    "finished_at": task_metadata["finished_at"],
                    "tool_calls": result_row.get("metrics", {}).get(
                        "n_tool_calls",
                        result_row.get("total_tool_calls"),
                    ),
                    "token_usage": result_row.get("token_usage"),
                }
                (trial_destination / "metadata.json").write_text(
                    json.dumps(
                        trial_metadata,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                normalized_values.append(normalized)
                raw_values.append(float(raw_score))
                starter_values.append(float(starter_score))
                reference_values.append(float(reference_score))
                trial_task_scores[trial][task] = normalized
                trial_records.append(
                    {
                        "trial": trial,
                        "normalized_score": normalized,
                        "raw_score": float(raw_score),
                        "score_path": _published_artifact_path(
                            trial_score_path,
                            output_parent=args.output.parent,
                        ),
                        "source_run_id": source_run,
                    }
                )

            task_scores[task] = statistics.fmean(normalized_values)
            task_score_stddevs[task] = statistics.stdev(normalized_values)
            raw_task_scores[task] = statistics.fmean(raw_values)
            raw_task_score_stddevs[task] = statistics.stdev(raw_values)
            task_anchors[task] = {
                "starter_score": statistics.fmean(starter_values),
                "reference_score": statistics.fmean(reference_values),
            }
            aggregate_score = {
                "benchmark": "learning_beyond_gradients_article_suite",
                "task": task,
                "trial_count": trial_count,
                "normalized_score": task_scores[task],
                "normalized_score_stddev": task_score_stddevs[task],
                "score": raw_task_scores[task],
                "score_stddev": raw_task_score_stddevs[task],
                "starter_score": task_anchors[task]["starter_score"],
                "reference_score": task_anchors[task]["reference_score"],
                "trials": trial_records,
            }
            aggregate_score_path = destination / "score.json"
            aggregate_score_path.write_text(
                json.dumps(aggregate_score, indent=2, sort_keys=True) + "\n"
            )
            metadata_payload = {
                "benchmark": "learning_beyond_gradients_article_suite",
                "task": task,
                "trial_count": trial_count,
                "current_task_digest": current_digest,
                "model": representative_metadata["model"],
                "harness": representative_metadata["harness"],
                "provider_reasoning_effort": representative_metadata[
                    "provider_reasoning_effort"
                ],
                "normalized_score": task_scores[task],
                "normalized_score_stddev": task_score_stddevs[task],
                "raw_score": raw_task_scores[task],
                "raw_score_stddev": raw_task_score_stddevs[task],
                "source_run_ids": [
                    record["source_run_id"] for record in trial_records
                ],
            }
            (destination / "metadata.json").write_text(
                json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n"
            )
            submission_details[task] = str(
                _published_artifact_path(
                    aggregate_score_path,
                    output_parent=args.output.parent,
                )
            )
            source_runs[task] = [
                record["source_run_id"] for record in trial_records
            ]
        aggregate_scores = _aggregate_task_scores(
            task_scores,
            trial_task_scores=trial_task_scores,
        )
        ranked.append(
            {
                "model_id": representative_metadata["model"]["id"],
                "model": representative_metadata["model"]["display_name"],
                "model_route": representative_metadata["model"]["model"],
                "provider": representative_metadata["model"]["provider"],
                "provider_label": PROVIDER_LABELS.get(
                    representative_metadata["model"]["provider"],
                    representative_metadata["model"]["provider"],
                ),
                "harness": representative_metadata["harness"],
                "provider_reasoning_effort": representative_metadata[
                    "provider_reasoning_effort"
                ],
                "trial_count": trial_count,
                **aggregate_scores,
                # Backward-compatible alias for consumers of the first
                # published schema. This is not the primary ranking metric.
                "average_normalized_score": aggregate_scores[
                    "arithmetic_mean_normalized_score"
                ],
                "task_scores": task_scores,
                "task_score_stddevs": task_score_stddevs,
                "raw_task_scores": raw_task_scores,
                "raw_task_score_stddevs": raw_task_score_stddevs,
                "task_anchors": task_anchors,
                "submission_details": submission_details,
                "source_runs": source_runs,
            }
        )
    ranked = _rank_rows(
        ranked,
        score_key="final_normalized_score",
    )
    leaderboards = _build_leaderboards(ranked)

    payload = {
        "benchmark": "learning_beyond_gradients_article_suite",
        "batch_id": (
            next(iter(model_batch_roots.values())).name
            if len(set(model_batch_roots.values())) == 1
            else "per-model-batches"
        ),
        "batch_ids": {
            model_id: batch_root.name
            for model_id, batch_root in model_batch_roots.items()
        },
        "protocol": protocol,
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
            "task_estimator": "mean_across_five_trials",
            "task_variability": "sample_standard_deviation",
            "trial_estimator": "iqm_across_nine_tasks",
            "final_estimator": "mean_across_five_trial_iqms",
            "final_variability": "sample_standard_deviation",
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
            row["model_id"]: row["source_runs"] for row in ranked
        },
        "generated_at": datetime.fromtimestamp(
            max(
                entry[0]["finished_at"]
                for model_results in task_results.values()
                for task_trials in model_results.values()
                for entry in task_trials.values()
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
