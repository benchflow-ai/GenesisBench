from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

import numpy as np

ATARI57_GAMES = (
    "Alien-v5",
    "Amidar-v5",
    "Assault-v5",
    "Asterix-v5",
    "Asteroids-v5",
    "Atlantis-v5",
    "BankHeist-v5",
    "BattleZone-v5",
    "BeamRider-v5",
    "Berzerk-v5",
    "Bowling-v5",
    "Boxing-v5",
    "Breakout-v5",
    "Centipede-v5",
    "ChopperCommand-v5",
    "CrazyClimber-v5",
    "Defender-v5",
    "DemonAttack-v5",
    "DoubleDunk-v5",
    "Enduro-v5",
    "FishingDerby-v5",
    "Freeway-v5",
    "Frostbite-v5",
    "Gopher-v5",
    "Gravitar-v5",
    "Hero-v5",
    "IceHockey-v5",
    "Jamesbond-v5",
    "Kangaroo-v5",
    "Krull-v5",
    "KungFuMaster-v5",
    "MontezumaRevenge-v5",
    "MsPacman-v5",
    "NameThisGame-v5",
    "Phoenix-v5",
    "Pitfall-v5",
    "Pong-v5",
    "PrivateEye-v5",
    "Qbert-v5",
    "Riverraid-v5",
    "RoadRunner-v5",
    "Robotank-v5",
    "Seaquest-v5",
    "Skiing-v5",
    "Solaris-v5",
    "SpaceInvaders-v5",
    "StarGunner-v5",
    "Surround-v5",
    "Tennis-v5",
    "TimePilot-v5",
    "Tutankham-v5",
    "UpNDown-v5",
    "Venture-v5",
    "VideoPinball-v5",
    "WizardOfWor-v5",
    "YarsRevenge-v5",
    "Zaxxon-v5",
)
OBSERVATION_MODES = ("ram", "native_obs")
SEARCH_REPEAT_INDICES = (0, 1, 2)
FRAME_BUDGET_PER_SEARCH = 20_000_000
ENVPOOL_VERSION = "1.1.1"
ARTICLE_SEARCH_EVIDENCE_FILES = (
    "policy.py",
    "trials.jsonl",
    "summary.csv",
    "sample_efficiency.png",
    "README.md",
)
ATARI_SETTINGS: dict[str, int | bool | float] = {
    "img_height": 210,
    "img_width": 160,
    "stack_num": 1,
    "gray_scale": False,
    "frame_skip": 1,
    "noop_max": 1,
    "use_fire_reset": True,
    "episodic_life": False,
    "reward_clip": False,
    "repeat_action_probability": 0.0,
    "full_action_space": False,
}


class Atari57ArtifactError(ValueError):
    """Raised when an aggregate Atari57 artifact violates the public contract."""


@dataclass(frozen=True)
class SearchTrajectory:
    env_id: str
    obs_mode: str
    repeat_index: int
    frame_budget: int = FRAME_BUDGET_PER_SEARCH


@dataclass(frozen=True)
class AtariPolicySpec:
    env_id: str
    obs_mode: str
    repeat_index: int
    module_path: Path
    config: dict[str, Any]


@dataclass(frozen=True)
class InteractionRecord:
    env_id: str
    obs_mode: str
    repeat_index: int
    cumulative_env_steps: int
    cumulative_episodes: int
    status: str
    evidence_path: Path | None


@dataclass(frozen=True)
class InteractionBudget:
    planned_trajectories: int
    completed_trajectories: int
    counted_env_steps: int
    target_env_steps: int


@dataclass(frozen=True)
class Atari57Artifact:
    root: Path
    manifest_path: Path
    policies: tuple[AtariPolicySpec, ...]
    interaction_records: tuple[InteractionRecord, ...]
    interaction_budget: InteractionBudget


@dataclass(frozen=True)
class AtariEpisode:
    env_id: str
    obs_mode: str
    repeat_index: int
    seed: int
    return_: float
    hns: float
    length: int
    terminated: bool
    truncated: bool
    invalid_action: bool
    policy_error: str | None
    mean_action_latency_ms: float


@dataclass(frozen=True)
class HNSReference:
    env_id: str
    known_best_score: float
    random_score: float
    human_score: float

    def normalize(self, score: float) -> float:
        denominator = self.human_score - self.random_score
        if denominator == 0:
            raise ValueError(f"Human and random anchors are equal for {self.env_id}")
        return (float(score) - self.random_score) / denominator


@dataclass(frozen=True)
class Atari57Evaluation:
    episodes: tuple[AtariEpisode, ...]
    per_game: dict[str, dict[str, Any]]
    score: float
    mean_best_input_mean_hns: float
    median_best_single_run_hns: float
    mean_best_single_run_hns: float

    @property
    def evaluation_trajectories(self) -> int:
        return len(self.episodes)

    @property
    def games_evaluated(self) -> int:
        return len(self.per_game)

    @property
    def invalid_episode_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return statistics.fmean(
            episode.invalid_action or episode.policy_error is not None
            for episode in self.episodes
        )

    @property
    def counted_evaluation_steps(self) -> int:
        return sum(episode.length for episode in self.episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "median_best_input_mean_hns": self.score,
            "mean_best_input_mean_hns": self.mean_best_input_mean_hns,
            "median_best_single_run_hns": self.median_best_single_run_hns,
            "mean_best_single_run_hns": self.mean_best_single_run_hns,
            "games_evaluated": self.games_evaluated,
            "evaluation_trajectories": self.evaluation_trajectories,
            "counted_evaluation_steps": self.counted_evaluation_steps,
            "invalid_episode_rate": self.invalid_episode_rate,
            "per_game": self.per_game,
            "episodes": [
                {
                    "env_id": episode.env_id,
                    "obs_mode": episode.obs_mode,
                    "repeat_index": episode.repeat_index,
                    "seed": episode.seed,
                    "return": episode.return_,
                    "hns": episode.hns,
                    "length": episode.length,
                    "terminated": episode.terminated,
                    "truncated": episode.truncated,
                    "invalid_action": episode.invalid_action,
                    "policy_error": episode.policy_error,
                    "mean_action_latency_ms": (episode.mean_action_latency_ms),
                }
                for episode in self.episodes
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def expected_search_trajectories() -> tuple[SearchTrajectory, ...]:
    return tuple(
        SearchTrajectory(
            env_id=env_id,
            obs_mode=obs_mode,
            repeat_index=repeat_index,
        )
        for env_id in ATARI57_GAMES
        for obs_mode in OBSERVATION_MODES
        for repeat_index in SEARCH_REPEAT_INDICES
    )


def load_hns_references(
    path: str | Path,
) -> dict[str, HNSReference]:
    table_path = Path(path)
    with table_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_fields = {
        "env_id",
        "known_best_score",
        "random_score",
        "human_score",
    }
    if not rows or not required_fields.issubset(rows[0]):
        raise ValueError(f"{table_path} must contain {sorted(required_fields)}")

    references: dict[str, HNSReference] = {}
    for row in rows:
        env_id = row["env_id"]
        if env_id not in ATARI57_GAMES:
            raise ValueError(f"Unknown Atari57 game in HNS table: {env_id}")
        if env_id in references:
            raise ValueError(f"Duplicate HNS row for {env_id}")
        reference = HNSReference(
            env_id=env_id,
            known_best_score=float(row["known_best_score"]),
            random_score=float(row["random_score"]),
            human_score=float(row["human_score"]),
        )
        if reference.human_score == reference.random_score:
            raise ValueError(f"Degenerate HNS anchors for {env_id}")
        references[env_id] = reference

    missing = set(ATARI57_GAMES) - set(references)
    if missing:
        raise ValueError(
            "HNS table is missing Atari57 games: " + ", ".join(sorted(missing))
        )
    return references


def aggregate_atari57_episodes(
    episodes: tuple[AtariEpisode, ...] | list[AtariEpisode],
) -> Atari57Evaluation:
    configured_episodes = tuple(episodes)
    if not configured_episodes:
        raise ValueError("At least one Atari episode is required")

    grouped: dict[tuple[str, str], list[AtariEpisode]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for episode in configured_episodes:
        if episode.env_id not in ATARI57_GAMES:
            raise ValueError(f"Unknown Atari57 game: {episode.env_id}")
        if episode.obs_mode not in OBSERVATION_MODES:
            raise ValueError(f"Unknown observation mode: {episode.obs_mode}")
        key = (episode.env_id, episode.obs_mode, episode.repeat_index)
        if key in seen:
            raise ValueError(f"Duplicate evaluation trajectory: {key}")
        seen.add(key)
        grouped[(episode.env_id, episode.obs_mode)].append(episode)

    games: dict[str, dict[str, Any]] = {}
    for env_id in ATARI57_GAMES:
        modes: dict[str, dict[str, Any]] = {}
        game_episodes: list[AtariEpisode] = []
        for obs_mode in OBSERVATION_MODES:
            mode_episodes = grouped.get((env_id, obs_mode), [])
            if not mode_episodes:
                continue
            game_episodes.extend(mode_episodes)
            modes[obs_mode] = {
                "mean_return": statistics.fmean(
                    episode.return_ for episode in mode_episodes
                ),
                "mean_hns": statistics.fmean(episode.hns for episode in mode_episodes),
                "repeat_count": len(mode_episodes),
            }
        if not modes:
            continue
        best_input_mode = max(
            modes,
            key=lambda mode: modes[mode]["mean_hns"],
        )
        games[env_id] = {
            "modes": modes,
            "best_input_mode": best_input_mode,
            "best_input_mean_hns": modes[best_input_mode]["mean_hns"],
            "best_single_run_hns": max(episode.hns for episode in game_episodes),
        }

    best_input_values = [game["best_input_mean_hns"] for game in games.values()]
    best_single_values = [game["best_single_run_hns"] for game in games.values()]
    return Atari57Evaluation(
        episodes=configured_episodes,
        per_game=games,
        score=float(statistics.median(best_input_values)),
        mean_best_input_mean_hns=statistics.fmean(best_input_values),
        median_best_single_run_hns=float(statistics.median(best_single_values)),
        mean_best_single_run_hns=statistics.fmean(best_single_values),
    )


def _load_policy_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"genesisbench_atari57_submission_{abs(hash(path.resolve()))}",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import policy from {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _call_with_supported_kwargs(callable_: Any, kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return callable_(**kwargs)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return callable_(**kwargs)
    supported = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return callable_(**supported)


def _instantiate_policy(
    module: ModuleType,
    *,
    spec: AtariPolicySpec,
    action_count: int,
    seed: int,
) -> Any:
    kwargs = {
        "env_id": spec.env_id,
        "obs_mode": spec.obs_mode,
        "repeat_index": spec.repeat_index,
        "action_count": action_count,
        "seed": seed,
        "config": dict(spec.config),
    }
    if hasattr(module, "make_policy"):
        return _call_with_supported_kwargs(module.make_policy, kwargs)
    if hasattr(module, "Policy"):
        return _call_with_supported_kwargs(module.Policy, kwargs)
    raise AttributeError(f"{spec.module_path} must define Policy or make_policy")


def _reset_policy(policy: Any, seed: int) -> None:
    reset = getattr(policy, "reset", None)
    if reset is not None:
        _call_with_supported_kwargs(reset, {"seed": seed})


def _policy_action(policy: Any, observation: Any, info: Any) -> Any:
    act = getattr(policy, "act", None)
    if act is None:
        raise AttributeError("Policy must define act")
    try:
        signature = inspect.signature(act)
    except (TypeError, ValueError):
        return act(observation, info)
    parameters = signature.parameters
    if "info" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return act(observation, info=info)
    positional = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    if len(positional) >= 2:
        return act(observation, info)
    return act(observation)


def _unbatch(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 0:
        return array.item()
    return value


def _unbatch_info(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: _unbatch(item) for key, item in value.items()}


def _parse_reset(value: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        return _unbatch(value[0]), _unbatch_info(value[1])
    return _unbatch(value), {}


def _parse_step(
    value: Any,
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    if not isinstance(value, tuple):
        raise TypeError("Environment step must return a tuple")
    if len(value) == 4:
        observation, reward, done, info = value
        return (
            _unbatch(observation),
            float(_unbatch(reward)),
            bool(_unbatch(done)),
            False,
            _unbatch_info(info),
        )
    if len(value) == 5:
        observation, reward, terminated, truncated, info = value
        return (
            _unbatch(observation),
            float(_unbatch(reward)),
            bool(_unbatch(terminated)),
            bool(_unbatch(truncated)),
            _unbatch_info(info),
        )
    raise TypeError("Environment step must return 4 (Gym) or 5 (Gymnasium) values")


def _action_count(action_space: Any) -> int:
    if hasattr(action_space, "n"):
        value = np.asarray(action_space.n)
        return int(value.reshape(-1)[0])
    if hasattr(action_space, "nvec"):
        value = np.asarray(action_space.nvec)
        return int(value.reshape(-1)[0])
    raise TypeError("Atari action space must expose n or nvec")


def _validate_action(action: Any, action_count: int) -> int:
    array = np.asarray(action)
    if array.size != 1:
        raise ValueError(f"Expected one discrete action, got shape {array.shape}")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, (np.floating, float)) and not float(scalar).is_integer():
        raise ValueError(f"Action must be an integer, got {scalar!r}")
    integer = int(scalar)
    if integer < 0 or integer >= action_count:
        raise ValueError(f"Action {integer} is outside [0, {action_count})")
    return integer


def _default_env_factory(
    env_id: str,
    seed: int,
    settings: dict[str, object],
) -> Any:
    try:
        import envpool
    except ImportError as error:
        raise RuntimeError(
            "EnvPool is required for Atari execution. Use the task Dockerfile "
            "or install envpool==1.1.1."
        ) from error
    if envpool.__version__ != ENVPOOL_VERSION:
        raise RuntimeError(
            f"Expected envpool=={ENVPOOL_VERSION}, got {envpool.__version__}"
        )
    return envpool.make_gym(
        env_id,
        num_envs=1,
        batch_size=1,
        seed=seed,
        **settings,
    )


EnvFactory = Callable[[str, int, dict[str, object]], Any]


class _DeterministicActionSpace:
    n = 4


class DeterministicAtariTestEnv:
    """Tiny deterministic vector environment used only for contract tests."""

    action_space = _DeterministicActionSpace()

    def __init__(self, env_id: str, seed: int) -> None:
        self.env_id = env_id
        self.seed = seed
        self.steps = 0

    def _observation(self) -> np.ndarray:
        game_index = ATARI57_GAMES.index(self.env_id)
        return np.asarray(
            [[game_index, self.seed % 997, self.steps]],
            dtype=np.int64,
        )

    def _info(self) -> dict[str, np.ndarray]:
        return {
            "ram": np.asarray(
                [[self.steps % 256, self.seed % 256]],
                dtype=np.uint8,
            )
        }

    def reset(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        self.steps = 0
        return self._observation(), self._info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, np.ndarray],
    ]:
        integer = int(np.asarray(action).reshape(-1)[0])
        self.steps += 1
        return (
            self._observation(),
            np.asarray([float(integer)]),
            np.asarray([False]),
            self._info(),
        )

    def close(self) -> None:
        return None


def deterministic_test_env_factory(
    env_id: str,
    seed: int,
    settings: dict[str, object],
) -> DeterministicAtariTestEnv:
    if settings != ATARI_SETTINGS:
        raise ValueError("Deterministic test backend requires fixed Atari settings")
    return DeterministicAtariTestEnv(env_id, seed)


def evaluate_atari57_artifact(
    artifact: str | Path | Atari57Artifact,
    *,
    games: Iterable[str],
    obs_modes: Iterable[str],
    seeds: Iterable[int],
    max_steps: int,
    hns_references: dict[str, HNSReference],
    env_factory: EnvFactory | None = None,
) -> Atari57Evaluation:
    configured_artifact = (
        artifact
        if isinstance(artifact, Atari57Artifact)
        else load_atari57_artifact(artifact)
    )
    configured_games = tuple(games)
    configured_modes = tuple(obs_modes)
    configured_seeds = tuple(seeds)
    if not configured_games:
        raise ValueError("At least one game is required")
    if not configured_modes:
        raise ValueError("At least one observation mode is required")
    if not configured_seeds:
        raise ValueError("At least one seed is required")
    if len(configured_seeds) > len(SEARCH_REPEAT_INDICES):
        raise ValueError("Atari57 evaluation supports at most three repeat seeds")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    for env_id in configured_games:
        if env_id not in ATARI57_GAMES:
            raise ValueError(f"Unknown Atari57 game: {env_id}")
        if env_id not in hns_references:
            raise ValueError(f"Missing HNS reference for {env_id}")
    for obs_mode in configured_modes:
        if obs_mode not in OBSERVATION_MODES:
            raise ValueError(f"Unknown observation mode: {obs_mode}")

    policies = {
        (policy.env_id, policy.obs_mode, policy.repeat_index): policy
        for policy in configured_artifact.policies
    }
    make_env = env_factory or _default_env_factory
    episodes: list[AtariEpisode] = []

    for env_id in configured_games:
        reference = hns_references[env_id]
        for obs_mode in configured_modes:
            for repeat_index, seed in enumerate(configured_seeds):
                policy_spec = policies[(env_id, obs_mode, repeat_index)]
                module = _load_policy_module(policy_spec.module_path)
                env = make_env(env_id, seed, dict(ATARI_SETTINGS))
                action_count = _action_count(env.action_space)
                policy = _instantiate_policy(
                    module,
                    spec=policy_spec,
                    action_count=action_count,
                    seed=seed,
                )
                _reset_policy(policy, seed)
                observation, info = _parse_reset(env.reset())
                episode_return = 0.0
                terminated = False
                truncated = False
                invalid_action = False
                policy_error: str | None = None
                latencies: list[float] = []
                length = 0

                try:
                    for _ in range(max_steps):
                        policy_info = (
                            {"ram": info["ram"]}
                            if obs_mode == "ram" and "ram" in info
                            else ({} if obs_mode == "ram" else None)
                        )
                        started = time.perf_counter()
                        try:
                            action = _policy_action(
                                policy,
                                observation,
                                policy_info,
                            )
                            action = _validate_action(
                                action,
                                action_count,
                            )
                        except Exception as error:
                            invalid_action = isinstance(error, ValueError)
                            policy_error = f"{type(error).__name__}: {error}"
                            episode_return = reference.random_score
                            break
                        finally:
                            latencies.append((time.perf_counter() - started) * 1000.0)

                        (
                            observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = _parse_step(env.step(np.asarray([action], dtype=np.int32)))
                        length += 1
                        episode_return += reward
                        if terminated or truncated:
                            break
                    else:
                        truncated = True
                finally:
                    close = getattr(env, "close", None)
                    if close is not None:
                        close()

                episodes.append(
                    AtariEpisode(
                        env_id=env_id,
                        obs_mode=obs_mode,
                        repeat_index=repeat_index,
                        seed=seed,
                        return_=episode_return,
                        hns=reference.normalize(episode_return),
                        length=length,
                        terminated=terminated,
                        truncated=truncated,
                        invalid_action=invalid_action,
                        policy_error=policy_error,
                        mean_action_latency_ms=(
                            statistics.fmean(latencies) if latencies else 0.0
                        ),
                    )
                )

    return aggregate_atari57_episodes(episodes)


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Atari57ArtifactError(f"{field} must be an object")
    return value


def _safe_relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Atari57ArtifactError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise Atari57ArtifactError(f"{field} must be a safe relative path")
    return path


def _resolve_under_root(root: Path, relative: Path, field: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise Atari57ArtifactError(f"{field} resolves outside the artifact")
    return resolved


def _load_manifest(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    candidate = Path(path).resolve()
    manifest_path = candidate / "manifest.json" if candidate.is_dir() else candidate
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise Atari57ArtifactError(
            f"manifest.json is not valid JSON: {error}"
        ) from error
    return manifest_path.parent, manifest_path, _mapping(manifest, "manifest")


def _validate_protocol(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != "1.0":
        raise Atari57ArtifactError("schema_version must be '1.0'")
    protocol = _mapping(manifest.get("protocol"), "protocol")
    expected = {
        "suite": "Atari57",
        "envpool_version": ENVPOOL_VERSION,
        "observation_modes": list(OBSERVATION_MODES),
        "search_repeats": len(SEARCH_REPEAT_INDICES),
        "frame_budget_per_search": FRAME_BUDGET_PER_SEARCH,
        "expected_search_trajectories": len(expected_search_trajectories()),
        "expected_policy_slots": len(expected_search_trajectories()),
        "expected_total_frame_target": (
            len(expected_search_trajectories()) * FRAME_BUDGET_PER_SEARCH
        ),
        "atari_settings": ATARI_SETTINGS,
    }
    for field, expected_value in expected.items():
        if protocol.get(field) != expected_value:
            raise Atari57ArtifactError(
                f"protocol.{field} must equal {expected_value!r}"
            )


def _policy_specs(
    root: Path,
    manifest: dict[str, Any],
) -> tuple[AtariPolicySpec, ...]:
    defaults = _mapping(manifest.get("policy_defaults"), "policy_defaults")
    overrides_value = manifest.get("policies", [])
    if not isinstance(overrides_value, list):
        raise Atari57ArtifactError("policies must be an array")

    overrides: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index, value in enumerate(overrides_value):
        override = _mapping(value, f"policies[{index}]")
        env_id = override.get("env_id")
        obs_mode = override.get("obs_mode")
        repeat_index = override.get("repeat_index")
        key = (env_id, obs_mode, repeat_index)
        if env_id not in ATARI57_GAMES:
            raise Atari57ArtifactError(f"policies[{index}].env_id is not in Atari57")
        if obs_mode not in OBSERVATION_MODES:
            raise Atari57ArtifactError(f"policies[{index}].obs_mode is invalid")
        if repeat_index not in SEARCH_REPEAT_INDICES:
            raise Atari57ArtifactError(f"policies[{index}].repeat_index is invalid")
        if key in overrides:
            raise Atari57ArtifactError(f"duplicate policy override for {key}")
        overrides[key] = override

    policies: list[AtariPolicySpec] = []
    for env_id in ATARI57_GAMES:
        for obs_mode in OBSERVATION_MODES:
            for repeat_index in SEARCH_REPEAT_INDICES:
                override = overrides.get((env_id, obs_mode, repeat_index), {})
                field = f"policy module for {env_id}/{obs_mode}/{repeat_index}"
                module = _safe_relative_path(
                    override.get("module", defaults.get("module")),
                    field,
                )
                module_path = _resolve_under_root(root, module, field)
                if not module_path.is_file():
                    raise Atari57ArtifactError(
                        f"policy module does not exist: {module}"
                    )
                default_config = defaults.get("config", {})
                override_config = override.get("config", {})
                config = {
                    **_mapping(default_config, "policy_defaults.config"),
                    **_mapping(
                        override_config,
                        f"policy config for {env_id}/{obs_mode}/{repeat_index}",
                    ),
                }
                policies.append(
                    AtariPolicySpec(
                        env_id=env_id,
                        obs_mode=obs_mode,
                        repeat_index=repeat_index,
                        module_path=module_path,
                        config=config,
                    )
                )
    return tuple(policies)


def _interaction_records(
    root: Path,
    manifest: dict[str, Any],
) -> tuple[InteractionRecord, ...]:
    ledger_path = _resolve_under_root(
        root,
        _safe_relative_path(
            manifest.get("interaction_ledger"),
            "interaction_ledger",
        ),
        "interaction_ledger",
    )
    if not ledger_path.is_file():
        raise Atari57ArtifactError(
            f"interaction ledger does not exist: {ledger_path.name}"
        )
    try:
        ledger = json.loads(ledger_path.read_text())
    except json.JSONDecodeError as error:
        raise Atari57ArtifactError(
            f"interaction ledger is not valid JSON: {error}"
        ) from error
    ledger_mapping = _mapping(ledger, "interaction ledger")
    if ledger_mapping.get("schema_version") != "1.0":
        raise Atari57ArtifactError("interaction ledger schema_version must be '1.0'")
    values = ledger_mapping.get("records")
    if not isinstance(values, list):
        raise Atari57ArtifactError("interaction ledger records must be an array")

    expected_keys = {
        (trajectory.env_id, trajectory.obs_mode, trajectory.repeat_index)
        for trajectory in expected_search_trajectories()
    }
    records: dict[tuple[str, str, int], InteractionRecord] = {}
    completed_evidence_paths: set[Path] = set()
    for index, value in enumerate(values):
        item = _mapping(value, f"interaction ledger records[{index}]")
        env_id = item.get("env_id")
        obs_mode = item.get("obs_mode")
        repeat_index = item.get("repeat_index")
        key = (env_id, obs_mode, repeat_index)
        if key not in expected_keys:
            raise Atari57ArtifactError(
                f"interaction ledger records[{index}] has invalid trajectory"
            )
        if key in records:
            raise Atari57ArtifactError(f"duplicate interaction ledger record for {key}")
        steps = item.get("cumulative_env_steps")
        episodes = item.get("cumulative_episodes")
        status = item.get("status")
        if not isinstance(steps, int) or steps < 0:
            raise Atari57ArtifactError(
                "cumulative_env_steps must be a non-negative integer"
            )
        if not isinstance(episodes, int) or episodes < 0:
            raise Atari57ArtifactError(
                "cumulative_episodes must be a non-negative integer"
            )
        if status not in {"not_run", "running", "complete", "failed"}:
            raise Atari57ArtifactError(
                "status must be not_run, running, complete, or failed"
            )
        if status == "complete" and steps < FRAME_BUDGET_PER_SEARCH:
            raise Atari57ArtifactError(
                "complete search trajectories must count at least "
                f"{FRAME_BUDGET_PER_SEARCH} environment steps"
            )
        evidence_path: Path | None = None
        evidence_value = item.get("evidence_path")
        if evidence_value is not None:
            evidence_path = _resolve_under_root(
                root,
                _safe_relative_path(
                    evidence_value,
                    f"interaction ledger records[{index}].evidence_path",
                ),
                f"interaction ledger records[{index}].evidence_path",
            )
        if status == "complete":
            missing = (
                list(ARTICLE_SEARCH_EVIDENCE_FILES)
                if evidence_path is None
                else [
                    filename
                    for filename in ARTICLE_SEARCH_EVIDENCE_FILES
                    if not (evidence_path / filename).is_file()
                    or not (evidence_path / filename)
                    .resolve()
                    .is_relative_to(root.resolve())
                ]
            )
            if missing:
                raise Atari57ArtifactError(
                    "complete search evidence is missing: " + ", ".join(missing)
                )
            if evidence_path in completed_evidence_paths:
                raise Atari57ArtifactError(
                    "complete search trajectories must use distinct evidence paths"
                )
            completed_evidence_paths.add(evidence_path)
        records[key] = InteractionRecord(
            env_id=env_id,
            obs_mode=obs_mode,
            repeat_index=repeat_index,
            cumulative_env_steps=steps,
            cumulative_episodes=episodes,
            status=status,
            evidence_path=evidence_path,
        )

    return tuple(records.values())


def load_atari57_artifact(path: str | Path) -> Atari57Artifact:
    root, manifest_path, manifest = _load_manifest(path)
    _validate_protocol(manifest)
    policies = _policy_specs(root, manifest)
    records = _interaction_records(root, manifest)
    expected_count = len(expected_search_trajectories())
    counted_steps = sum(record.cumulative_env_steps for record in records)
    budget = InteractionBudget(
        planned_trajectories=expected_count,
        completed_trajectories=sum(record.status == "complete" for record in records),
        counted_env_steps=counted_steps,
        target_env_steps=expected_count * FRAME_BUDGET_PER_SEARCH,
    )
    return Atari57Artifact(
        root=root,
        manifest_path=manifest_path,
        policies=policies,
        interaction_records=records,
        interaction_budget=budget,
    )
