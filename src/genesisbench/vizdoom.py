from __future__ import annotations

import ast
import json
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import numpy as np

from genesisbench.policy_isolation import (
    close_policy,
    instantiate_policy,
    load_policy_module,
)

D1_ALLOWED_VARIABLES = ("HEALTH",)
D3_ALLOWED_VARIABLES = (
    "HEALTH",
    "AMMO2",
    "HITCOUNT",
    "DAMAGECOUNT",
    "KILLCOUNT",
)
VIZDOOM_ARTICLE_ENVPOOL_VERSION = "1.1.1"

_FORBIDDEN_IMPORT_ROOTS = {"envpool", "vizdoom"}
_FORBIDDEN_POLICY_NAMES = {
    "automap",
    "cfg_path",
    "env_id",
    "game_state",
    "label_buffer",
    "labels",
    "linedefs",
    "map_id",
    "object_positions",
    "sectors",
    "things",
    "vertices",
    "wad_path",
}
_FORBIDDEN_STRING_FRAGMENTS = (
    ".wad",
    "d1_basic.cfg",
    "d3_battle.cfg",
    "/oracle",
    "/verifier",
)


class PolicySourceViolation(ValueError):
    """Raised when a policy tries to access privileged VizDoom state."""


@dataclass(frozen=True)
class VizDoomEpisode:
    seed: int
    lane: int
    return_: float
    length: int
    terminated: bool
    truncated: bool
    invalid_action: bool
    policy_error: str | None
    mean_action_latency_ms: float
    final_variables: dict[str, float]


@dataclass(frozen=True)
class VizDoomEvaluation:
    scenario: str
    policy_path: str
    envpool_version: str
    batch_seed: int
    max_steps: int
    frame_skip: int
    render_width: int
    render_height: int
    episodes: tuple[VizDoomEpisode, ...]

    @property
    def mean_return(self) -> float:
        return float(np.mean([episode.return_ for episode in self.episodes]))

    @property
    def min_return(self) -> float:
        return float(np.min([episode.return_ for episode in self.episodes]))

    @property
    def max_return(self) -> float:
        return float(np.max([episode.return_ for episode in self.episodes]))

    @property
    def invalid_episode_rate(self) -> float:
        return float(
            np.mean(
                [
                    episode.invalid_action or episode.policy_error is not None
                    for episode in self.episodes
                ]
            )
        )

    @property
    def mean_action_latency_ms(self) -> float:
        return float(
            np.mean(
                [episode.mean_action_latency_ms for episode in self.episodes]
            )
        )

    def to_dict(self) -> dict[str, Any]:
        rendered_episodes = []
        for episode in self.episodes:
            rendered_episode = asdict(episode)
            rendered_episode["return"] = rendered_episode.pop("return_")
            rendered_episodes.append(rendered_episode)
        return {
            "scenario": self.scenario,
            "policy_path": self.policy_path,
            "envpool_version": self.envpool_version,
            "batch_seed": self.batch_seed,
            "max_steps": self.max_steps,
            "frame_skip": self.frame_skip,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "score": self.mean_return,
            "mean_return": self.mean_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "invalid_episode_rate": self.invalid_episode_rate,
            "mean_action_latency_ms": self.mean_action_latency_ms,
            "episodes": rendered_episodes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _policy_files(policy_path: Path) -> tuple[Path, ...]:
    root = policy_path.parent
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise PolicySourceViolation(
                f"Policy bundle may not contain symlinks: {candidate}"
            )
    return tuple(sorted(root.rglob("*.py")))


def _import_root(alias: ast.alias) -> str:
    return alias.name.split(".", 1)[0].lower()


def audit_vizdoom_policy(policy_path: str | Path) -> None:
    """Reject direct access to maps, labels, objects, or EnvPool internals."""

    path = Path(policy_path).resolve()
    if path.is_dir():
        path = path / "policy.py"
    if not path.is_file():
        raise FileNotFoundError(path)

    for source_path in _policy_files(path):
        try:
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
        except (OSError, SyntaxError) as error:
            raise PolicySourceViolation(
                f"Unable to audit {source_path.name}: {error}"
            ) from error

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {_import_root(alias) for alias in node.names}
                forbidden = roots & _FORBIDDEN_IMPORT_ROOTS
                if forbidden:
                    raise PolicySourceViolation(
                        "Policy may not import simulator internals: "
                        + ", ".join(sorted(forbidden))
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0].lower()
                if root in _FORBIDDEN_IMPORT_ROOTS:
                    raise PolicySourceViolation(
                        f"Policy may not import simulator internals: {root}"
                    )
            elif isinstance(node, ast.Name):
                if node.id.lower() in _FORBIDDEN_POLICY_NAMES:
                    raise PolicySourceViolation(
                        f"Forbidden privileged-state identifier: {node.id}"
                    )
            elif isinstance(node, ast.Attribute):
                if node.attr.lower() in _FORBIDDEN_POLICY_NAMES:
                    raise PolicySourceViolation(
                        f"Forbidden privileged-state attribute: {node.attr}"
                    )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.lower()
                if any(
                    fragment in value
                    for fragment in _FORBIDDEN_STRING_FRAGMENTS
                ):
                    raise PolicySourceViolation(
                        "Policy source references a forbidden map or hidden path"
                    )


def _load_policy_module(policy_path: Path) -> ModuleType:
    module_name = f"genesisbench_vizdoom_submission_{abs(hash(policy_path))}"
    return load_policy_module(policy_path, module_name=module_name)


def _instantiate_policy(module: ModuleType) -> Any:
    return instantiate_policy(module, init_kwargs={})


def _reset_policy(policy: Any) -> None:
    reset = getattr(policy, "reset", None)
    if reset is not None:
        reset()


def _readonly_frame(frame: np.ndarray) -> np.ndarray:
    public_frame = np.array(frame, dtype=np.uint8, copy=True, order="C")
    public_frame.flags.writeable = False
    return public_frame


def _to_hwc(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"Expected a three-dimensional frame, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected channel-last image data, got {array.shape}")
    return array


def _row_variables(
    info: Mapping[str, Any],
    row: int,
    allowed_variables: Iterable[str],
) -> dict[str, float]:
    variables: dict[str, float] = {}
    for name in allowed_variables:
        if name not in info:
            raise KeyError(f"EnvPool did not provide required variable {name}")
        values = np.asarray(info[name])
        variables[name] = float(values[row])
    return variables


def _active_info(
    info: Mapping[str, Any],
    keep: np.ndarray,
    row_count: int,
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value)[keep]
        for key, value in info.items()
        if np.asarray(value).ndim > 0
        and len(np.asarray(value)) == row_count
    }


def _validate_d1_action(action: Any) -> int:
    array = np.asarray(action)
    if array.shape != ():
        raise ValueError(f"Expected a scalar D1 action, got {array.shape}")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError("Boolean actions are not valid D1 actions")
    value = float(array)
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(f"Expected an integer D1 action, got {action!r}")
    integer = int(value)
    if integer < 0 or integer > 5:
        raise ValueError(f"D1 action must be in [0, 5], got {integer}")
    return integer


def _validate_d3_action(action: Any) -> np.ndarray:
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (8,):
        raise ValueError(f"Expected D3 action shape (8,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("D3 action contains NaN or infinite values")
    if np.any(array[:7] < 0.0) or np.any(array[:7] > 1.0):
        raise ValueError("D3 button channels must be in [0, 1]")
    if abs(float(array[7])) > 12.0:
        raise ValueError("D3 turn delta must be in [-12, 12]")
    return array


def _make_d1_env(
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    frame_skip: int,
    render_width: int,
    render_height: int,
) -> tuple[Any, None]:
    import envpool

    _require_article_envpool_version(envpool)
    env = envpool.make_gymnasium(
        "D1Basic-v1",
        num_envs=episodes,
        seed=seed,
        use_combined_action=True,
        stack_num=1,
        frame_skip=frame_skip,
        max_episode_steps=max_steps,
        render_mode="rgb_array",
        render_width=render_width,
        render_height=render_height,
    )
    return env, None


def _d3_config_text(*, width: int, height: int) -> str:
    from envpool.vizdoom.registration import maps_path

    source = Path(maps_path) / "D3_battle.cfg"
    config = source.read_text()
    replacements = {
        "screen_resolution = RES_160X120": (
            f"screen_resolution = RES_{width}X{height}"
        ),
        "screen_format = GRAY8": "screen_format = CRCGCB",
        "render_weapon = true": "render_weapon = false",
        "render_crosshair = true": "render_crosshair = false",
    }
    for original, replacement in replacements.items():
        if original not in config:
            raise RuntimeError(
                f"EnvPool D3 config is missing expected line: {original}"
            )
        config = config.replace(original, replacement, 1)

    buttons = """available_buttons =
    {
        ATTACK
        SPEED
        MOVE_FORWARD
        MOVE_BACKWARD
        MOVE_RIGHT
        MOVE_LEFT
        TURN180
        TURN_LEFT_RIGHT_DELTA
    }

"""
    start = config.index("available_buttons")
    end = config.index("# Game variables")
    return config[:start] + buttons + config[end:]


def _require_article_envpool_version(envpool: ModuleType) -> None:
    version = getattr(envpool, "__version__", None)
    if version != VIZDOOM_ARTICLE_ENVPOOL_VERSION:
        raise RuntimeError(
            "VizDoom article tasks require EnvPool "
            f"{VIZDOOM_ARTICLE_ENVPOOL_VERSION}, found {version!r}"
        )


def _make_d3_env(
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    frame_skip: int,
    render_width: int,
    render_height: int,
) -> tuple[Any, tempfile.TemporaryDirectory[str]]:
    import envpool

    _require_article_envpool_version(envpool)
    temporary_directory = tempfile.TemporaryDirectory(
        prefix="genesisbench-simulation-heuristics-vizdoom-d3-v1-"
    )
    config_path = Path(temporary_directory.name) / "battle-screen-cv.cfg"
    config_path.write_text(
        _d3_config_text(width=render_width, height=render_height)
    )
    try:
        env = envpool.make_gymnasium(
            "D3Battle-v1",
            num_envs=episodes,
            seed=seed,
            cfg_path=str(config_path),
            use_combined_action=False,
            stack_num=1,
            frame_skip=frame_skip,
            max_episode_steps=max_steps,
            img_width=render_width,
            img_height=render_height,
            reward_config={
                "DAMAGECOUNT": [1.0, 0.0],
                "KILLCOUNT": [10.0, 0.0],
            },
            selected_weapon_reward_config={},
        )
    except Exception:
        temporary_directory.cleanup()
        raise
    return env, temporary_directory


def _failed_evaluation(
    *,
    scenario: str,
    policy_path: Path,
    seed: int,
    episodes: int,
    max_steps: int,
    frame_skip: int,
    render_width: int,
    render_height: int,
    failure_return: float,
    error: Exception,
) -> VizDoomEvaluation:
    message = f"{type(error).__name__}: {error}"
    failed = tuple(
        VizDoomEpisode(
            seed=seed + lane,
            lane=lane,
            return_=failure_return,
            length=0,
            terminated=False,
            truncated=False,
            invalid_action=isinstance(error, ValueError),
            policy_error=message,
            mean_action_latency_ms=0.0,
            final_variables={},
        )
        for lane in range(episodes)
    )
    return VizDoomEvaluation(
        scenario=scenario,
        policy_path=str(policy_path),
        envpool_version=VIZDOOM_ARTICLE_ENVPOOL_VERSION,
        batch_seed=seed,
        max_steps=max_steps,
        frame_skip=frame_skip,
        render_width=render_width,
        render_height=render_height,
        episodes=failed,
    )


def evaluate_vizdoom_policy(
    policy_path: str | Path,
    *,
    scenario: str,
    seed: int,
    episodes: int = 10,
    max_steps: int | None = None,
    frame_skip: int | None = None,
    render_width: int | None = None,
    render_height: int | None = None,
    failure_return: float = 0.0,
) -> VizDoomEvaluation:
    """Evaluate a screen-only D1 or D3 policy in one EnvPool seed batch."""

    if scenario not in {"d1", "d3"}:
        raise ValueError(f"Unsupported VizDoom scenario: {scenario}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    defaults = {
        "d1": (2100, 1, 240, 180),
        "d3": (1050, 2, 640, 480),
    }
    default_steps, default_skip, default_width, default_height = defaults[scenario]
    configured_steps = default_steps if max_steps is None else max_steps
    configured_skip = default_skip if frame_skip is None else frame_skip
    configured_width = default_width if render_width is None else render_width
    configured_height = (
        default_height if render_height is None else render_height
    )

    path = Path(policy_path).resolve()
    if path.is_dir():
        path = path / "policy.py"
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        audit_vizdoom_policy(path)
        module = _load_policy_module(path)
    except Exception as error:
        return _failed_evaluation(
            scenario=scenario,
            policy_path=path,
            seed=seed,
            episodes=episodes,
            max_steps=configured_steps,
            frame_skip=configured_skip,
            render_width=configured_width,
            render_height=configured_height,
            failure_return=failure_return,
            error=error,
        )

    make_env = _make_d1_env if scenario == "d1" else _make_d3_env
    env, temporary_directory = make_env(
        episodes=episodes,
        seed=seed,
        max_steps=configured_steps,
        frame_skip=configured_skip,
        render_width=configured_width,
        render_height=configured_height,
    )
    allowed_variables = (
        D1_ALLOWED_VARIABLES
        if scenario == "d1"
        else D3_ALLOWED_VARIABLES
    )
    validate_action = (
        _validate_d1_action if scenario == "d1" else _validate_d3_action
    )

    returns = np.zeros(episodes, dtype=np.float64)
    lengths = np.zeros(episodes, dtype=np.int64)
    terminated_by_lane = np.zeros(episodes, dtype=np.bool_)
    truncated_by_lane = np.zeros(episodes, dtype=np.bool_)
    invalid_by_lane = np.zeros(episodes, dtype=np.bool_)
    errors: list[str | None] = [None] * episodes
    latencies: list[list[float]] = [[] for _ in range(episodes)]
    final_variables: list[dict[str, float]] = [
        {} for _ in range(episodes)
    ]
    policies: dict[int, Any] = {}

    try:
        observations, info = env.reset()
        active_ids = np.asarray(info["env_id"], dtype=np.int64)
        active_info: dict[str, Any] = dict(info)
        active_observations = np.asarray(observations)
        for lane in range(episodes):
            try:
                policy = _instantiate_policy(module)
                _reset_policy(policy)
                policies[lane] = policy
            except Exception as error:
                returns[lane] = failure_return
                errors[lane] = f"{type(error).__name__}: {error}"

        rendered_frames = None
        if scenario == "d1":
            rendered_frames = env.render(
                env_ids=np.arange(episodes, dtype=np.int64)
            )

        for _ in range(configured_steps):
            step_ids: list[int] = []
            actions: list[Any] = []

            for row, lane_value in enumerate(active_ids):
                lane = int(lane_value)
                if lane not in policies or errors[lane] is not None:
                    continue
                variables = _row_variables(
                    active_info,
                    row,
                    allowed_variables,
                )
                final_variables[lane] = variables
                if scenario == "d1":
                    assert rendered_frames is not None
                    frame = rendered_frames[lane]
                else:
                    frame = _to_hwc(active_observations[row])

                started_at = time.perf_counter()
                try:
                    public_variables = MappingProxyType(dict(variables))
                    action = policies[lane].act(
                        _readonly_frame(frame),
                        public_variables,
                    )
                    action = validate_action(action)
                except Exception as error:
                    errors[lane] = f"{type(error).__name__}: {error}"
                    invalid_by_lane[lane] = isinstance(error, ValueError)
                    returns[lane] = failure_return
                    continue
                finally:
                    latencies[lane].append(
                        (time.perf_counter() - started_at) * 1000.0
                    )
                step_ids.append(lane)
                actions.append(action)

            if not step_ids:
                break

            action_array = np.asarray(
                actions,
                dtype=np.int64 if scenario == "d1" else np.float64,
            )
            next_observations, rewards, terminated, truncated, info = env.step(
                action_array,
                np.asarray(step_ids, dtype=np.int64),
            )
            done = np.logical_or(terminated, truncated)
            current_ids = np.asarray(info["env_id"], dtype=np.int64)

            for row, lane_value in enumerate(current_ids):
                lane = int(lane_value)
                returns[lane] += float(np.asarray(rewards)[row])
                lengths[lane] += 1
                final_variables[lane] = _row_variables(
                    info,
                    row,
                    allowed_variables,
                )
                if bool(done[row]):
                    terminated_by_lane[lane] = bool(terminated[row])
                    truncated_by_lane[lane] = bool(truncated[row])

            keep = ~done
            active_ids = current_ids[keep]
            active_info = _active_info(info, keep, len(done))
            if scenario == "d3":
                active_observations = np.asarray(next_observations)[keep]
            if len(active_ids) == 0:
                break
            if scenario == "d1":
                rendered_frames = env.render(
                    env_ids=np.arange(episodes, dtype=np.int64)
                )

        unfinished = {
            int(lane)
            for lane in active_ids
            if errors[int(lane)] is None
        }
        for lane in unfinished:
            truncated_by_lane[lane] = True
    finally:
        for policy in policies.values():
            close_policy(policy)
        env.close()
        if temporary_directory is not None:
            temporary_directory.cleanup()

    episode_results = tuple(
        VizDoomEpisode(
            seed=seed + lane,
            lane=lane,
            return_=float(returns[lane]),
            length=int(lengths[lane]),
            terminated=bool(terminated_by_lane[lane]),
            truncated=bool(truncated_by_lane[lane]),
            invalid_action=bool(invalid_by_lane[lane]),
            policy_error=errors[lane],
            mean_action_latency_ms=(
                float(np.mean(latencies[lane])) if latencies[lane] else 0.0
            ),
            final_variables=final_variables[lane],
        )
        for lane in range(episodes)
    )
    return VizDoomEvaluation(
        scenario=scenario,
        policy_path=str(path),
        envpool_version=VIZDOOM_ARTICLE_ENVPOOL_VERSION,
        batch_seed=seed,
        max_steps=configured_steps,
        frame_skip=configured_skip,
        render_width=configured_width,
        render_height=configured_height,
        episodes=episode_results,
    )
