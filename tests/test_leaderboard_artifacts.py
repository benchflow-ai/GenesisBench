from __future__ import annotations

import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = REPO_ROOT / "leaderboard" / "simulation_heuristics_ant_v1.json"
ARTICLE_SUITE = REPO_ROOT / "leaderboard" / "article_suite.json"
ARTICLE_SUITE_MARKDOWN = REPO_ROOT / "leaderboard" / "ARTICLE_SUITE.md"
ROOT_README = REPO_ROOT / "README.md"


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
    assert payload["leaderboards"][-1]["id"] == "average"

    averages = [
        row["average_normalized_score"] for row in payload["rows"]
    ]
    assert averages == sorted(averages, reverse=True)

    for row in payload["rows"]:
        assert row["harness"] == "opencode"
        assert set(row["task_scores"]) == set(payload["tasks"])
        assert set(row["submission_details"]) == set(payload["tasks"])
        assert math.isclose(
            row["average_normalized_score"],
            sum(row["task_scores"].values()) / payload["task_count"],
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
        scores = [row["normalized_score"] for row in board["rows"]]
        assert scores == sorted(scores, reverse=True)
        assert len(board["rows"]) == len(payload["rows"])
        for row in board["rows"]:
            assert row["normalized_score"] == model_rows[row["model_id"]][
                "task_scores"
            ][board["id"]]

    average_board = payload["leaderboards"][-1]
    assert [
        row["average_normalized_score"] for row in average_board["rows"]
    ] == averages

    markdown = ARTICLE_SUITE_MARKDOWN.read_text()
    assert markdown.count("| Rank | Model | Harness | Effort |") == 10
    headings = [
        line for line in markdown.splitlines() if line.startswith("## ")
    ]
    assert len(headings) == 10
    assert headings[-1] == "## 10. Nine-task average"

    root_readme = ROOT_README.read_text()
    assert "| Rank | Model | Nine-task average |" in root_readme
    assert "## Legacy Ant-Only Leaderboard" not in root_readme
