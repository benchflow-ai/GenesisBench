from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tomllib

import pytest
import yaml

from genesisbench.breakout import _validate_action, evaluate_breakout_policy


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "tasks"
TASKS = {
    "ram": TASK_ROOT / "simulation_heuristics_breakout_ram_v1",
    "rgb": TASK_ROOT / "simulation_heuristics_breakout_rgb_v1",
}


def _envpool_runtime_available() -> bool:
    try:
        import envpool

        env = envpool.make_gym(
            "Breakout-v5",
            num_envs=1,
            batch_size=1,
            seed=0,
            max_episode_steps=2,
            img_height=210,
            img_width=160,
            stack_num=1,
            gray_scale=False,
            frame_skip=1,
            noop_max=1,
            use_fire_reset=True,
            episodic_life=False,
            reward_clip=False,
            repeat_action_probability=0.0,
            full_action_space=False,
        )
        close = getattr(env, "close", None)
        if close is not None:
            close()
    except Exception:
        return False
    return True


ENVPOOL_RUNTIME_AVAILABLE = _envpool_runtime_available()
requires_envpool = pytest.mark.skipif(
    not ENVPOOL_RUNTIME_AVAILABLE,
    reason="envpool==1.1.1 is not installed on this platform",
)


def _task_config(task_dir: Path) -> dict:
    document = task_dir.joinpath("task.md").read_text()
    _, front_matter, _ = document.split("---", 2)
    return yaml.safe_load(front_matter)


def _load_hidden_evaluator(task_dir: Path):
    path = task_dir / "verifier" / "evaluate_hidden.py"
    name = f"test_hidden_{task_dir.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("mode", "task_name", "development_steps"),
    (
        ("ram", "genesisbench/simulation_heuristics_breakout_ram_v1", 27000),
        ("rgb", "genesisbench/simulation_heuristics_breakout_rgb_v1", 30000),
    ),
)
def test_native_packages_declare_separate_observation_contracts(
    mode: str,
    task_name: str,
    development_steps: int,
) -> None:
    task_dir = TASKS[mode]
    config = _task_config(task_dir)

    assert config["task"]["name"] == task_name
    assert config["metadata"]["genesisbench"]["development"] == {
        "episodes": 1,
        "max_steps": development_steps,
        "seeds": [0],
        "observation_mode": mode,
    }
    assert config["metadata"]["genesisbench"]["submission"] == {
        "directory": "final_policy",
        "entrypoint": "policy.py",
    }

    required = (
        "README.md",
        "task.md",
        "evaluate.py",
        "environment/Dockerfile",
        "starter_policy/policy.py",
        "oracle/policy.py",
        "oracle/solve.sh",
        "task_context/evaluation.md",
        "task_context/article_progression.md",
        "task_context/policy_api.md",
        "task_context/provenance.md",
        "verifier/anchors.json",
        "verifier/config.toml",
        "verifier/evaluate_hidden.py",
        "verifier/test.sh",
        "verifier/verifier.md",
        "verifier/rubrics/verifier.md",
        "verifier/rubrics/verifier.toml",
    )
    assert all(task_dir.joinpath(path).is_file() for path in required)
    assert "cp -R /app/starter_policy /app/final_policy" in task_dir.joinpath(
        "environment/Dockerfile"
    ).read_text()
    dockerfile = task_dir.joinpath("environment/Dockerfile").read_text()
    assert '"envpool==1.1.1"' in dockerfile
    assert "AutoROM" not in dockerfile


def test_runtime_uses_article_envpool_configuration() -> None:
    source = REPO_ROOT.joinpath("src/genesisbench/breakout.py").read_text()
    for snippet in (
        'version("envpool")',
        '"Breakout-v5"',
        "frame_skip=1",
        "noop_max=1",
        "use_fire_reset=True",
        "episodic_life=False",
        "reward_clip=False",
        "repeat_action_probability=variant.repeat_action_probability",
        "full_action_space=False",
    ):
        assert snippet in source


def test_action_validation_accepts_only_reduced_breakout_actions() -> None:
    assert [_validate_action(action) for action in (0, 1, 2, 3)] == [
        0,
        1,
        2,
        3,
    ]
    with pytest.raises(ValueError):
        _validate_action(4)
    with pytest.raises(ValueError):
        _validate_action(1.5)
    with pytest.raises(ValueError):
        _validate_action([1, 2])


@requires_envpool
@pytest.mark.parametrize(
    ("mode", "expected_shape"),
    (("ram", (128,)), ("rgb", (3, 210, 160))),
)
def test_evaluator_passes_only_the_declared_observation(
    tmp_path: Path,
    mode: str,
    expected_shape: tuple[int, ...],
) -> None:
    policy = tmp_path / f"{mode}_policy.py"
    policy.write_text(
        "import numpy as np\n"
        f"EXPECTED = {expected_shape!r}\n"
        "class Policy:\n"
        "    def reset(self, seed=0):\n"
        "        self.calls = 0\n"
        "    def act(self, observation):\n"
        "        value = np.asarray(observation)\n"
        "        assert value.shape == EXPECTED\n"
        "        assert value.dtype == np.uint8\n"
        "        self.calls += 1\n"
        "        return 0\n"
    )

    result = evaluate_breakout_policy(
        policy,
        observation_mode=mode,
        seeds=(7,),
        max_steps=2,
    )

    assert result.episodes[0].length == 2
    assert result.invalid_episode_rate == 0.0


@requires_envpool
@pytest.mark.parametrize("mode", ("ram", "rgb"))
def test_starter_policy_short_smoke(mode: str) -> None:
    result = evaluate_breakout_policy(
        TASKS[mode] / "starter_policy" / "policy.py",
        observation_mode=mode,
        seeds=(0,),
        max_steps=10,
    )

    assert result.episodes[0].length == 10
    assert result.invalid_episode_rate == 0.0
    assert result.mean_return == 0.0


@requires_envpool
def test_rgb_intermediate_reproduces_428(tmp_path: Path) -> None:
    starter_source = TASKS["rgb"].joinpath("starter_policy/policy.py").read_text()
    assert "CHASE_LEAD_STEPS = 6.0" in starter_source
    intermediate = tmp_path / "rgb_428_policy.py"
    intermediate.write_text(
        starter_source.replace(
            "CHASE_LEAD_STEPS = 6.0",
            "CHASE_LEAD_STEPS = 8.0",
            1,
        )
    )

    result = evaluate_breakout_policy(
        intermediate,
        observation_mode="rgb",
        seeds=(0,),
        max_steps=30000,
    )

    assert result.mean_return == 428.0


@requires_envpool
@pytest.mark.parametrize(
    ("mode", "starter_nominal"),
    (("ram", 387.0), ("rgb", 310.0)),
)
def test_anchor_normalization_and_article_target(
    mode: str,
    starter_nominal: float,
) -> None:
    task_dir = TASKS[mode]
    hidden = _load_hidden_evaluator(task_dir)
    evaluation = tomllib.loads(
        task_dir.joinpath("verifier/config.toml").read_text()
    )["evaluation"]
    anchors = json.loads(task_dir.joinpath("verifier/anchors.json").read_text())

    starter_path = task_dir / "verifier" / anchors["starter_policy"]["path"]
    reference_path = task_dir / "verifier" / anchors["reference_policy"]["path"]
    assert starter_path.read_bytes() == task_dir.joinpath(
        "starter_policy/policy.py"
    ).read_bytes()
    assert reference_path.read_bytes() == task_dir.joinpath(
        "oracle/policy.py"
    ).read_bytes()

    starter_suites, starter_score = hidden._evaluate_raw(
        starter_path,
        evaluation=evaluation,
    )
    reference_suites, reference_score = hidden._evaluate_raw(
        reference_path,
        evaluation=evaluation,
    )

    assert starter_score < reference_score
    assert reference_score == 864.0
    assert starter_suites["hidden_nominal"].mean_return == starter_nominal
    assert all(
        episode.return_ == 864.0
        for result in reference_suites.values()
        for episode in result.episodes
    )
    assert 100.0 * (starter_score - starter_score) / (
        reference_score - starter_score
    ) == 0.0
    assert 100.0 * (reference_score - starter_score) / (
        reference_score - starter_score
    ) == 100.0
    assert starter_suites
