from __future__ import annotations

import importlib.util
import inspect
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Literal

import numpy as np


ObservationMode = Literal["ram", "rgb"]


@dataclass(frozen=True)
class BreakoutVariant:
    name: str = "nominal"
    repeat_action_probability: float = 0.0
    initial_actions: tuple[int, ...] = ()


@dataclass(frozen=True)
class BreakoutEpisode:
    seed: int
    variant: str
    observation_mode: ObservationMode
    return_: float
    length: int
    warmup_steps: int
    final_lives: int
    terminated: bool
    truncated: bool
    invalid_action: bool
    policy_error: str | None
    mean_action_latency_ms: float


@dataclass(frozen=True)
class BreakoutEvaluation:
    policy_path: str
    observation_mode: ObservationMode
    max_steps: int
    episodes: tuple[BreakoutEpisode, ...]

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
    def completion_rate(self) -> float:
        return float(
            np.mean([episode.return_ >= 864.0 for episode in self.episodes])
        )

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
            "policy_path": self.policy_path,
            "observation_mode": self.observation_mode,
            "max_steps": self.max_steps,
            "score": self.mean_return,
            "mean_return": self.mean_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "completion_rate": self.completion_rate,
            "invalid_episode_rate": self.invalid_episode_rate,
            "mean_action_latency_ms": self.mean_action_latency_ms,
            "episodes": rendered_episodes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _load_policy_module(policy_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"genesisbench_breakout_submission_{abs(hash(policy_path.resolve()))}",
        policy_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import policy from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _instantiate_policy(module: ModuleType, seed: int) -> Any:
    if hasattr(module, "make_policy"):
        factory = module.make_policy
        try:
            return factory(seed=seed)
        except TypeError:
            return factory()
    if hasattr(module, "Policy"):
        policy_class = module.Policy
        try:
            return policy_class(seed=seed)
        except TypeError:
            return policy_class()
    raise AttributeError("Submission must define Policy or make_policy")


def _reset_policy(policy: Any, seed: int) -> None:
    reset = getattr(policy, "reset", None)
    if reset is None:
        return
    try:
        signature = inspect.signature(reset)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "seed" in signature.parameters:
        reset(seed=seed)
    else:
        reset()


def _validate_action(action: Any) -> int:
    array = np.asarray(action)
    if array.shape == (1,):
        array = array.reshape(())
    if array.shape != ():
        raise ValueError(f"Expected a scalar action, got shape {array.shape}")
    value = float(array)
    if not np.isfinite(value):
        raise ValueError("Action is NaN or infinite")
    if not value.is_integer():
        raise ValueError(f"Action must be an integer, got {value}")
    action_id = int(value)
    if action_id not in (0, 1, 2, 3):
        raise ValueError(f"Action must be one of 0, 1, 2, 3; got {action_id}")
    return action_id


def _make_breakout_env(
    *,
    variant: BreakoutVariant,
    max_steps: int,
    seed: int,
) -> Any:
    try:
        import envpool
        from importlib.metadata import version
    except ImportError as error:
        raise RuntimeError(
            "Breakout evaluation requires envpool==1.1.1"
        ) from error
    installed_version = version("envpool")
    if installed_version != "1.1.1":
        raise RuntimeError(
            f"Breakout evaluation requires envpool==1.1.1, got {installed_version}"
        )

    env = envpool.make_gym(
        "Breakout-v5",
        num_envs=1,
        batch_size=1,
        seed=seed,
        max_episode_steps=max_steps + len(variant.initial_actions),
        img_height=210,
        img_width=160,
        stack_num=1,
        gray_scale=False,
        frame_skip=1,
        noop_max=1,
        use_fire_reset=True,
        episodic_life=False,
        reward_clip=False,
        repeat_action_probability=variant.repeat_action_probability,
        full_action_space=False,
    )
    if int(env.action_space.n) != 4:
        raise RuntimeError(f"Expected four Breakout actions, got {env.action_space}")
    return env


def _reset_env(env: Any) -> tuple[np.ndarray, dict[str, Any]]:
    result = env.reset()
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("EnvPool 1.1.1 reset must return (observation, info)")
    observation, info = result
    return np.asarray(observation), info


def _step_env(
    env: Any,
    action: int,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    result = env.step(np.asarray([action], dtype=np.int32))
    if len(result) != 5:
        raise RuntimeError("EnvPool 1.1.1 step must return five values")
    observation, reward, terminated, truncated, info = result
    return (
        np.asarray(observation),
        float(np.asarray(reward)[0]),
        bool(np.asarray(terminated)[0]),
        bool(np.asarray(truncated)[0]),
        info,
    )


def _policy_observation(
    observation: np.ndarray,
    info: dict[str, Any],
    mode: ObservationMode,
) -> np.ndarray:
    if mode == "ram":
        ram = np.asarray(info["ram"], dtype=np.uint8)
        if ram.shape != (1, 128):
            raise RuntimeError(f"Unexpected EnvPool RAM shape: {ram.shape}")
        return ram[0].copy()

    pixels = np.asarray(observation, dtype=np.uint8)
    if pixels.shape != (1, 3, 210, 160):
        raise RuntimeError(f"Unexpected EnvPool RGB shape: {pixels.shape}")
    return pixels[0].copy()


def _info_lives(info: dict[str, Any]) -> int:
    lives = np.asarray(info.get("lives", (0,)))
    return int(lives.reshape(-1)[0]) if lives.size else 0


def _apply_initial_actions(
    env: Any,
    observation: np.ndarray,
    info: dict[str, Any],
    actions: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any], float]:
    warmup_return = 0.0
    for action in actions:
        action_id = _validate_action(action)
        observation, reward, terminated, truncated, info = _step_env(
            env,
            action_id,
        )
        warmup_return += reward
        if terminated or truncated:
            raise RuntimeError(
                "Breakout variant terminated during its initial action prefix"
            )
    return observation, info, warmup_return


def evaluate_breakout_policy(
    policy_path: str | Path,
    *,
    observation_mode: ObservationMode,
    seeds: Iterable[int],
    max_steps: int = 108000,
    variants: Iterable[BreakoutVariant] | None = None,
    failure_return: float = -1.0,
) -> BreakoutEvaluation:
    path = Path(policy_path).resolve()
    if path.is_dir():
        path = path / "policy.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    if observation_mode not in ("ram", "rgb"):
        raise ValueError(f"Unsupported observation mode: {observation_mode}")

    module = _load_policy_module(path)
    configured_variants = tuple(variants or (BreakoutVariant(),))
    configured_seeds = tuple(int(seed) for seed in seeds)
    episodes: list[BreakoutEpisode] = []

    for variant in configured_variants:
        for seed in configured_seeds:
            env = _make_breakout_env(
                variant=variant,
                max_steps=max_steps,
                seed=seed,
            )
            observation, info = _reset_env(env)
            observation, info, episode_return = _apply_initial_actions(
                env,
                observation,
                info,
                variant.initial_actions,
            )
            policy = _instantiate_policy(module, seed)
            _reset_policy(policy, seed)
            terminated = False
            truncated = False
            invalid_action = False
            policy_error: str | None = None
            latencies: list[float] = []
            length = 0

            try:
                for length in range(1, max_steps + 1):
                    started_at = time.perf_counter()
                    try:
                        action = _validate_action(
                            policy.act(
                                _policy_observation(
                                    observation,
                                    info,
                                    observation_mode,
                                )
                            )
                        )
                    except Exception as error:
                        policy_error = f"{type(error).__name__}: {error}"
                        invalid_action = isinstance(error, ValueError)
                        episode_return = failure_return
                        break
                    latencies.append((time.perf_counter() - started_at) * 1000)
                    observation, reward, terminated, truncated, info = _step_env(
                        env,
                        action,
                    )
                    episode_return += reward
                    if terminated or truncated:
                        break
            finally:
                close = getattr(env, "close", None)
                if close is not None:
                    close()

            episodes.append(
                BreakoutEpisode(
                    seed=seed,
                    variant=variant.name,
                    observation_mode=observation_mode,
                    return_=float(episode_return),
                    length=length,
                    warmup_steps=len(variant.initial_actions),
                    final_lives=_info_lives(info),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    invalid_action=invalid_action,
                    policy_error=policy_error,
                    mean_action_latency_ms=float(np.mean(latencies))
                    if latencies
                    else 0.0,
                )
            )

    return BreakoutEvaluation(
        policy_path=str(path),
        observation_mode=observation_mode,
        max_steps=max_steps,
        episodes=tuple(episodes),
    )
