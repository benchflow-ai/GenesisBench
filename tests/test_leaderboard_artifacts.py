from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = REPO_ROOT / "leaderboard" / "simulation_heuristics_ant_v1.json"


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
