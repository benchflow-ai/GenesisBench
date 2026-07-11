from __future__ import annotations

from pathlib import Path

from genesisbench.ant import DynamicsVariant, evaluate_ant_policy


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER = REPO_ROOT / "tasks" / "ant_v1" / "starter_policy" / "policy.py"


def test_starter_policy_short_smoke() -> None:
    result = evaluate_ant_policy(
        STARTER,
        seeds=(0,),
        max_steps=10,
    )

    assert len(result.episodes) == 1
    assert result.episodes[0].length == 10
    assert result.invalid_episode_rate == 0.0
    assert result.mean_return > 0.0


def test_dynamics_variant_short_smoke() -> None:
    result = evaluate_ant_policy(
        STARTER,
        seeds=(1,),
        max_steps=5,
        variants=(
            DynamicsVariant(
                name="test_variant",
                density_scale=1.05,
                friction_scale=0.95,
                damping_scale=1.05,
                actuator_scale=0.95,
            ),
        ),
    )

    assert result.episodes[0].variant == "test_variant"
    assert result.invalid_episode_rate == 0.0


def test_json_uses_public_return_key() -> None:
    result = evaluate_ant_policy(
        STARTER,
        seeds=(2,),
        max_steps=2,
    )

    episode = result.to_dict()["episodes"][0]
    assert "return" in episode
    assert "return_" not in episode

