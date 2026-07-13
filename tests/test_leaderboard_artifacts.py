from __future__ import annotations

import json
import math
from pathlib import Path
import struct


REPO_ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = REPO_ROOT / "leaderboard" / "simulation_heuristics_ant_v1.json"
ARTICLE_SUITE = REPO_ROOT / "leaderboard" / "article_suite.json"
ARTICLE_SUITE_MARKDOWN = REPO_ROOT / "leaderboard" / "ARTICLE_SUITE.md"
ROOT_README = REPO_ROOT / "README.md"
WEBSITE = REPO_ROOT / "website" / "index.html"
WEBSITE_DATA = REPO_ROOT / "website" / "assets" / "article_suite.json"
TASK_IMAGE = REPO_ROOT / "leaderboard" / "article_suite_task_leaderboards.png"
FINAL_IMAGE = REPO_ROOT / "leaderboard" / "article_suite_final_leaderboard.png"


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


def test_packaged_leaderboard_is_self_contained() -> None:
    payload = json.loads(LEADERBOARD.read_text())
    assert payload["benchmark"] == "simulation_heuristics_ant_v1"
    assert len(payload["rows"]) == 4

    for row in payload["rows"]:
        assert "run_dir" not in row
        assert not Path(row["submission_path"]).is_absolute()

        submission = REPO_ROOT / row["submission_path"]
        score_path = submission.parent / "score.json"
        metadata_path = submission.parent / "metadata.json"
        assert submission.is_file()
        assert score_path.is_file()
        assert metadata_path.is_file()

        score = json.loads(score_path.read_text())
        for suite in ("hidden_nominal", "hidden_robustness"):
            policy_path = score[suite]["policy_path"]
            assert policy_path == row["submission_path"]
            assert not Path(policy_path).is_absolute()


def test_article_suite_leaderboard_is_complete_and_self_contained() -> None:
    payload = json.loads(ARTICLE_SUITE.read_text())

    assert payload["benchmark"] == "learning_beyond_gradients_article_suite"
    assert payload["task_count"] == 9
    assert payload["leaderboard_count"] == 10
    assert len(payload["tasks"]) == 9
    assert len(payload["rows"]) == 4
    assert [row["rank"] for row in payload["rows"]] == [1, 2, 3, 4]
    assert [board["id"] for board in payload["leaderboards"][:-1]] == payload[
        "tasks"
    ]
    assert payload["leaderboards"][-1]["id"] == "final"
    assert payload["aggregation"]["primary_metric"] == "interquartile_mean"
    assert payload["aggregation"]["trim_fraction_per_tail"] == 0.25
    assert payload["aggregation"]["trimmed_score_count_per_tail"] == 2
    assert payload["aggregation"]["retained_score_count"] == 5
    assert (
        payload["inference_settings"]["cross_provider_comparability"]
        == "labels_are_not_a_shared_numeric_compute_scale"
    )
    inference_by_model = {
        row["model_id"]: row for row in payload["inference_settings"]["models"]
    }
    assert inference_by_model["gpt-5.6-sol"]["provider_reasoning_effort"] == "max"
    assert inference_by_model["gpt-5.5"]["provider_reasoning_effort"] == "xhigh"
    assert (
        inference_by_model["claude-opus-4.8"]["provider_reasoning_effort"]
        == "max"
    )
    assert (
        inference_by_model["gpt-5.4-mini"]["provider_reasoning_effort"]
        == "xhigh"
    )

    final_scores = [
        row["final_normalized_score"] for row in payload["rows"]
    ]
    assert final_scores == sorted(final_scores, reverse=True)

    for row in payload["rows"]:
        assert row["harness"] == "opencode"
        assert set(row["task_scores"]) == set(payload["tasks"])
        assert set(row["raw_task_scores"]) == set(payload["tasks"])
        assert set(row["task_anchors"]) == set(payload["tasks"])
        assert set(row["submission_details"]) == set(payload["tasks"])
        sorted_scores = sorted(row["task_scores"].values())
        expected_iqm = sum(sorted_scores[2:7]) / 5
        assert math.isclose(
            row["final_normalized_score"],
            expected_iqm,
        )
        assert math.isclose(
            row["arithmetic_mean_normalized_score"],
            sum(row["task_scores"].values()) / payload["task_count"],
        )
        assert (
            row["average_normalized_score"]
            == row["arithmetic_mean_normalized_score"]
        )
        for task, relative_score_path in row["submission_details"].items():
            score_path = REPO_ROOT / relative_score_path
            metadata_path = score_path.with_name("metadata.json")
            assert not Path(relative_score_path).is_absolute()
            assert score_path.is_file()
            assert metadata_path.is_file()

            score = json.loads(score_path.read_text())
            metadata = json.loads(metadata_path.read_text())
            assert metadata["task"] == task
            assert metadata["harness"] == "opencode"
            assert metadata["normalized_score"] == row["task_scores"][task]
            assert score["normalized_score"] == row["task_scores"][task]

            def assert_no_absolute_artifact_paths(value: object) -> None:
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {"policy_path", "artifact_path"}:
                            assert item == "submitted_artifact"
                        assert_no_absolute_artifact_paths(item)
                elif isinstance(value, list):
                    for item in value:
                        assert_no_absolute_artifact_paths(item)

            assert_no_absolute_artifact_paths(score)

    model_rows = {row["model_id"]: row for row in payload["rows"]}
    for board in payload["leaderboards"][:-1]:
        scores = [row["raw_score"] for row in board["rows"]]
        assert scores == sorted(scores, reverse=True)
        assert len(board["rows"]) == len(payload["rows"])
        for row in board["rows"]:
            model_row = model_rows[row["model_id"]]
            assert row["normalized_score"] == model_row["task_scores"][
                board["id"]
            ]
            assert row["raw_score"] == model_row["raw_task_scores"][
                board["id"]
            ]
            assert row["starter_score"] == model_row["task_anchors"][
                board["id"]
            ]["starter_score"]
            assert row["reference_score"] == model_row["task_anchors"][
                board["id"]
            ]["reference_score"]

    final_board = payload["leaderboards"][-1]
    assert [
        row["final_normalized_score"] for row in final_board["rows"]
    ] == final_scores
    assert final_board["display_transform"]["offset"] == 100.0
    assert all(
        math.isclose(
            row["positive_display_score"],
            row["final_normalized_score"] + 100.0,
        )
        for row in final_board["rows"]
    )
    assert [row["model"] for row in payload["rows"]] == [
        "GPT-5.5",
        "GPT-5.6 Sol",
        "Claude Opus 4.8",
        "GPT-5.4 Mini",
    ]

    markdown = ARTICLE_SUITE_MARKDOWN.read_text()
    assert "article_suite_task_leaderboards.png" in markdown
    assert "article_suite_final_leaderboard.png" in markdown
    assert "| Rank |" not in markdown

    root_readme = ROOT_README.read_text()
    assert "leaderboard/article_suite_final_leaderboard.png" in root_readme
    assert "leaderboard/article_suite_task_leaderboards.png" not in root_readme
    assert "| Rank | Model | Nine-task average |" not in root_readme
    assert "## Legacy Ant-Only Leaderboard" not in root_readme

    website = WEBSITE.read_text()
    assert "assets/article_suite_task_leaderboards.png" not in website
    assert "assets/article_suite_final_leaderboard.png" not in website
    assert '<h3 id="leaderboardsTitle">Leaderboards</h3>' in website
    assert "Task-level view" not in website
    assert 'id="taskLeaderboards"' in website
    assert 'id="finalLeaderboard"' in website
    assert 'id="inferenceSettings"' in website
    assert "not a shared numeric compute scale" in website
    assert 'fetch("assets/article_suite.json")' in website
    assert json.loads(WEBSITE_DATA.read_text()) == payload

    for source in (TASK_IMAGE, FINAL_IMAGE):
        assert source.is_file()
        width, height = _png_dimensions(source)
        assert width >= 1500
        assert height >= 800
