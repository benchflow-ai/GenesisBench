from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from genesisbench.integrity import run_integrity_audit
from scripts.run_article_suite import _opencode_config


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "tasks"


def _write_trajectory(path: Path, events: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n"
    )


def test_clean_submission_and_trajectory_pass(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "policy.py").write_text(
        "import math\n\nclass Policy:\n"
        "    def act(self, observation):\n"
        "        return math.floor(float(observation[0]))\n"
    )
    trajectory = tmp_path / "trajectory.jsonl"
    _write_trajectory(
        trajectory,
        [
            {"type": "user_message", "text": "task"},
            {
                "type": "tool_call",
                "kind": "read",
                "title": "read",
                "arguments": {"filePath": "/app/task_context/evaluation.md"},
            },
        ],
    )
    report = run_integrity_audit(
        submission=submission,
        trajectory=trajectory,
        config={"require_trajectory": True},
    )
    assert report.passed
    assert report.submission_sha256


def test_webfetch_and_forbidden_path_force_failure(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "policy.py").write_text(
        "from pathlib import Path\n"
        "SECRET = Path('/verifier/config.toml')\n"
        "class Policy:\n"
        "    def act(self, observation): return 0\n"
    )
    trajectory = tmp_path / "trajectory.jsonl"
    _write_trajectory(
        trajectory,
        [
            {
                "type": "tool_call",
                "kind": "fetch",
                "title": "webfetch",
                "arguments": {"url": "https://example.com/answer.py"},
            }
        ],
    )
    report = run_integrity_audit(
        submission=submission,
        trajectory=trajectory,
        config={"require_trajectory": True},
    )
    codes = {violation.code for violation in report.violations}
    assert not report.passed
    assert "RH001_FORBIDDEN_PATH" in codes
    assert "RH002_EXTERNAL_NETWORK_TOOL" in codes


def test_forbidden_answer_hash_fails_but_oracle_is_allowed(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    payload = b"class Policy:\n    def act(self, observation): return 0\n"
    (submission / "policy.py").write_bytes(payload)
    forbidden_hash = hashlib.sha256(payload).hexdigest()
    trajectory = tmp_path / "trajectory.jsonl"
    _write_trajectory(trajectory, [{"type": "agent_message", "text": "done"}])

    failed = run_integrity_audit(
        submission=submission,
        trajectory=trajectory,
        config={
            "require_trajectory": True,
            "forbidden_source_sha256": [forbidden_hash],
        },
    )
    assert "RH009_FORBIDDEN_SOURCE_HASH" in {
        violation.code for violation in failed.violations
    }

    _write_trajectory(
        trajectory,
        [{"type": "oracle", "command": "bash /oracle/solve.sh", "return_code": 0}],
    )
    passed = run_integrity_audit(
        submission=submission,
        trajectory=trajectory,
        config={
            "require_trajectory": True,
            "forbidden_source_sha256": [forbidden_hash],
        },
    )
    assert passed.passed


def test_symlink_submission_fails(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    target = tmp_path / "target.py"
    target.write_text("print('unsafe')\n")
    (submission / "policy.py").symlink_to(target)
    trajectory = tmp_path / "trajectory.jsonl"
    _write_trajectory(trajectory, [{"type": "agent_message", "text": "done"}])
    report = run_integrity_audit(
        submission=submission,
        trajectory=trajectory,
        config={"require_trajectory": True},
    )
    assert "RH005_SYMLINK" in {
        violation.code for violation in report.violations
    }


def test_untrusted_completed_search_ledger_fails(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "policy.py").write_text(
        "class Policy:\n    def act(self, observation): return 0\n"
    )
    (submission / "interaction_ledger.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "env_id": "Pong-v5",
                        "obs_mode": "ram",
                        "repeat_index": 0,
                        "status": "complete",
                        "cumulative_env_steps": 20_000_000,
                    }
                ]
            }
        )
    )
    trajectory = tmp_path / "trajectory.jsonl"
    _write_trajectory(trajectory, [{"type": "agent_message", "text": "done"}])
    report = run_integrity_audit(
        submission=submission,
        trajectory=trajectory,
        config={
            "require_trajectory": True,
            "reject_untrusted_complete_ledger": True,
        },
    )
    assert "RH010_UNTRUSTED_EXPERIMENT_ACCOUNTING" in {
        violation.code for violation in report.violations
    }


def test_all_article_tasks_are_offline_and_isolated() -> None:
    task_paths = sorted(TASKS_ROOT.glob("simulation_heuristics_*/task.md"))
    assert len(task_paths) == 9
    for task_path in task_paths:
        frontmatter = yaml.safe_load(task_path.read_text().split("---", 2)[1])
        assert frontmatter["agent"]["user"] == "agent"
        assert frontmatter["agent"]["network_mode"] == "no-network"
        assert frontmatter["verifier"]["user"] == "root"
        assert frontmatter["verifier"]["network_mode"] == "no-network"
        assert frontmatter["verifier"]["hardening"] == {
            "cleanup_conftests": True
        }
        assert frontmatter["environment"]["network_mode"] == "no-network"
        assert frontmatter["environment"]["allow_internet"] is False

        task_dir = task_path.parent
        dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
        assert "security/restrict_exec.c" in dockerfile
        assert 'GENESISBENCH_POLICY_ISOLATION="required"' in dockerfile
        assert "/app/work" in dockerfile
        assert (task_dir / "verifier" / "integrity.json").is_file()
        verifier_script = (task_dir / "verifier" / "test.sh").read_text()
        assert "genesisbench.integrity" in verifier_script
        assert "GENESISBENCH_POLICY_ISOLATION=required" in verifier_script


def test_article_runner_denies_external_tools() -> None:
    config = _opencode_config(
        {
            "model": "azure/test-model",
            "display_name": "Test",
            "provider": "azure",
            "provider_reasoning_effort": "xhigh",
        }
    )
    assert config["permission"] == {
        "external_directory": "deny",
        "task": "deny",
        "webfetch": "deny",
        "websearch": "deny",
    }


def test_required_isolation_fails_closed_off_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from genesisbench import policy_isolation

    monkeypatch.setenv("GENESISBENCH_POLICY_ISOLATION", "required")
    monkeypatch.setattr(policy_isolation.sys, "platform", "darwin")
    with pytest.raises(policy_isolation.PolicyIsolationError):
        policy_isolation.isolation_enabled()
