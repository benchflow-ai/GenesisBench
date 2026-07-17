from __future__ import annotations

import inspect
import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from genesisbench.policy_isolation import (
    close_policy,
    instantiate_policy,
    load_policy_module,
)

PONG_TARGET_SCORE = 21.0


@dataclass(frozen=True)
class PongVariant:
    """Trusted Atari settings for one Pong evaluation condition."""

    name: str = "nominal"
    noop_max: int = 1
    frame_skip: int = 1
    repeat_action_probability: float = 0.0
    use_fire_reset: bool = True


@dataclass(frozen=True)
class PongEpisode:
    seed: int
    variant: str
    score: float
    length: int
    points_for: int
    points_against: int
    terminated: bool
    truncated: bool
    invalid_action: bool
    policy_error: str | None
    mean_action_latency_ms: float

    @property
    def won(self) -> bool:
        return self.score > 0.0

    @property
    def target_reached(self) -> bool:
        return math.isclose(self.score, PONG_TARGET_SCORE)


@dataclass(frozen=True)
class PongEvaluation:
    policy_path: str
    max_steps: int
    episodes: tuple[PongEpisode, ...]

    @property
    def mean_score(self) -> float:
        return float(np.mean([episode.score for episode in self.episodes]))

    @property
    def min_score(self) -> float:
        return float(np.min([episode.score for episode in self.episodes]))

    @property
    def max_score(self) -> float:
        return float(np.max([episode.score for episode in self.episodes]))

    @property
    def win_rate(self) -> float:
        return float(np.mean([episode.won for episode in self.episodes]))

    @property
    def target_score_rate(self) -> float:
        return float(np.mean([episode.target_reached for episode in self.episodes]))

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
            np.mean([episode.mean_action_latency_ms for episode in self.episodes])
        )

    def to_dict(self) -> dict[str, Any]:
        rendered_episodes = []
        for episode in self.episodes:
            rendered = asdict(episode)
            rendered["won"] = episode.won
            rendered["target_reached"] = episode.target_reached
            rendered_episodes.append(rendered)
        return {
            "policy_path": self.policy_path,
            "max_steps": self.max_steps,
            "target_score": PONG_TARGET_SCORE,
            "score": self.mean_score,
            "mean_score": self.mean_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "win_rate": self.win_rate,
            "target_score_rate": self.target_score_rate,
            "invalid_episode_rate": self.invalid_episode_rate,
            "mean_action_latency_ms": self.mean_action_latency_ms,
            "episodes": rendered_episodes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


EnvironmentFactory = Callable[[int, PongVariant], Any]


def _load_policy_module(policy_path: Path) -> ModuleType:
    module_name = f"genesisbench_pong_submission_{abs(hash(policy_path.resolve()))}"
    return load_policy_module(policy_path, module_name=module_name)


def _instantiate_policy(module: ModuleType, seed: int) -> Any:
    return instantiate_policy(module, init_kwargs={"seed": seed})


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


def _make_pong_env(seed: int, variant: PongVariant) -> Any:
    try:
        import envpool
    except ImportError as error:
        raise RuntimeError(
            "Pong evaluation requires envpool==1.1.1. "
            "Use the task environment or install that package locally."
        ) from error

    return envpool.make_gym(
        "Pong-v5",
        num_envs=1,
        batch_size=1,
        seed=seed,
        img_height=210,
        img_width=160,
        stack_num=1,
        gray_scale=True,
        frame_skip=variant.frame_skip,
        noop_max=variant.noop_max,
        use_fire_reset=variant.use_fire_reset,
        episodic_life=False,
        reward_clip=False,
        repeat_action_probability=variant.repeat_action_probability,
        full_action_space=False,
    )


def _reset_env(environment: Any) -> tuple[Any, dict[str, Any]]:
    result = environment.reset()
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("Pong environment reset did not return (obs, info)")
    observation, info = result
    if not isinstance(info, dict):
        raise RuntimeError("Pong environment reset info must be a mapping")
    return observation, info


def _step_env(
    environment: Any,
    action: int,
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    result = environment.step(np.asarray([action], dtype=np.int32))
    if not isinstance(result, tuple):
        raise RuntimeError("Pong environment step returned an invalid result")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    elif len(result) == 4:
        observation, reward, done, info = result
        terminated = done
        truncated = False
    else:
        raise RuntimeError(f"Pong environment step returned {len(result)} values")
    return (
        observation,
        float(np.asarray(reward).reshape(-1)[0]),
        bool(np.asarray(terminated).reshape(-1)[0]),
        bool(np.asarray(truncated).reshape(-1)[0]),
        info,
    )


def _extract_ram(info: dict[str, Any]) -> np.ndarray:
    if "ram" not in info:
        raise RuntimeError("Pong environment did not expose info['ram']")
    ram = np.asarray(info["ram"])
    if ram.shape == (1, 128):
        ram = ram[0]
    if ram.shape != (128,):
        raise RuntimeError(f"Expected Pong RAM shape (128,), got {ram.shape}")
    return np.asarray(ram, dtype=np.uint8).copy()


def _validate_action(action: Any, action_count: int) -> int:
    array = np.asarray(action)
    if array.shape not in ((), (1,)):
        raise ValueError(f"Expected one discrete Pong action, got shape {array.shape}")
    value = array.reshape(-1)[0] if array.shape else array.item()
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Pong action must be numeric, got {value!r}") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"Pong action must be a finite integer, got {value!r}")
    integer = int(numeric)
    if not 0 <= integer < action_count:
        raise ValueError(
            f"Pong action must be in [0, {action_count - 1}], got {integer}"
        )
    return integer


def _failure_episode(
    *,
    seed: int,
    variant: PongVariant,
    failure_score: float,
    length: int,
    error: Exception,
    invalid_action: bool,
    latencies: list[float],
) -> PongEpisode:
    return PongEpisode(
        seed=seed,
        variant=variant.name,
        score=failure_score,
        length=length,
        points_for=0,
        points_against=21,
        terminated=False,
        truncated=False,
        invalid_action=invalid_action,
        policy_error=f"{type(error).__name__}: {error}",
        mean_action_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
    )


def evaluate_pong_policy(
    policy_path: str | Path,
    *,
    seeds: Iterable[int],
    max_steps: int = 27_000,
    variants: Iterable[PongVariant] | None = None,
    failure_score: float = -21.0,
    environment_factory: EnvironmentFactory | None = None,
) -> PongEvaluation:
    """Evaluate a policy using only the current 128-byte Pong RAM state."""

    path = Path(policy_path).resolve()
    if path.is_dir():
        path = path / "policy.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    module = _load_policy_module(path)
    configured_seeds = tuple(int(seed) for seed in seeds)
    if not configured_seeds:
        raise ValueError("At least one evaluation seed is required")
    configured_variants = tuple(variants or (PongVariant(),))
    if not configured_variants:
        raise ValueError("At least one Pong variant is required")
    make_environment = environment_factory or _make_pong_env
    episodes: list[PongEpisode] = []

    for variant in configured_variants:
        for seed in configured_seeds:
            environment = make_environment(seed, variant)
            latencies: list[float] = []
            length = 0
            policy: Any | None = None
            try:
                try:
                    policy = _instantiate_policy(module, seed)
                    _reset_policy(policy, seed)
                    _, info = _reset_env(environment)
                except Exception as error:
                    episodes.append(
                        _failure_episode(
                            seed=seed,
                            variant=variant,
                            failure_score=failure_score,
                            length=0,
                            error=error,
                            invalid_action=False,
                            latencies=latencies,
                        )
                    )
                    continue

                score = 0.0
                points_for = 0
                points_against = 0
                terminated = False
                truncated = False
                failed_episode: PongEpisode | None = None

                for length in range(1, max_steps + 1):
                    started_at = time.perf_counter()
                    try:
                        ram = _extract_ram(info)
                        proposed_action = policy.act(ram)
                    except Exception as error:
                        failed_episode = _failure_episode(
                            seed=seed,
                            variant=variant,
                            failure_score=failure_score,
                            length=length,
                            error=error,
                            invalid_action=False,
                            latencies=latencies,
                        )
                        break
                    latencies.append((time.perf_counter() - started_at) * 1000.0)
                    try:
                        action = _validate_action(
                            proposed_action,
                            int(environment.action_space.n),
                        )
                    except Exception as error:
                        failed_episode = _failure_episode(
                            seed=seed,
                            variant=variant,
                            failure_score=failure_score,
                            length=length,
                            error=error,
                            invalid_action=True,
                            latencies=latencies,
                        )
                        break

                    _, reward, terminated, truncated, info = _step_env(
                        environment,
                        action,
                    )
                    score += reward
                    if reward > 0.0:
                        points_for += round(reward)
                    elif reward < 0.0:
                        points_against += round(-reward)
                    if terminated or truncated:
                        break

                if failed_episode is not None:
                    episodes.append(failed_episode)
                    continue
                if length >= max_steps and not terminated and not truncated:
                    truncated = True
                episodes.append(
                    PongEpisode(
                        seed=seed,
                        variant=variant.name,
                        score=float(score),
                        length=length,
                        points_for=points_for,
                        points_against=points_against,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        invalid_action=False,
                        policy_error=None,
                        mean_action_latency_ms=float(np.mean(latencies))
                        if latencies
                        else 0.0,
                    )
                )
            finally:
                if policy is not None:
                    close_policy(policy)
                close = getattr(environment, "close", None)
                if close is not None:
                    close()

    return PongEvaluation(
        policy_path=str(path),
        max_steps=max_steps,
        episodes=tuple(episodes),
    )
