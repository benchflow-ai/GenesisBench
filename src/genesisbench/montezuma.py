from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np


MONTEZUMA_ENV_ID = "MontezumaRevenge-v5"
MONTEZUMA_ACTION_COUNT = 18


@dataclass(frozen=True)
class MontezumaVariant:
    name: str = "canonical"
    bootstrap_steps: int = 0
    pre_policy_noops: int = 0

    @property
    def is_recovery(self) -> bool:
        return self.bootstrap_steps > 0 or self.pre_policy_noops > 0


@dataclass(frozen=True)
class MontezumaEpisode:
    seed: int
    variant: str
    bootstrap_steps: int
    pre_policy_noops: int
    return_: float
    length: int
    policy_steps: int
    target_reached: bool
    terminated: bool
    truncated: bool
    invalid_action: bool
    policy_error: str | None
    mean_action_latency_ms: float


@dataclass(frozen=True)
class MontezumaEvaluation:
    policy_path: str
    max_steps: int
    target_score: float
    episodes: tuple[MontezumaEpisode, ...]

    @property
    def mean_return(self) -> float:
        return float(np.mean([episode.return_ for episode in self.episodes]))

    @property
    def capped_mean_score(self) -> float:
        return float(
            np.mean(
                [
                    np.clip(episode.return_, 0.0, self.target_score)
                    for episode in self.episodes
                ]
            )
        )

    @property
    def target_success_rate(self) -> float:
        return float(np.mean([episode.target_reached for episode in self.episodes]))

    @property
    def recovery_success_rate(self) -> float:
        recovery_episodes = [
            episode
            for episode in self.episodes
            if episode.bootstrap_steps > 0 or episode.pre_policy_noops > 0
        ]
        if not recovery_episodes:
            return 0.0
        return float(np.mean([episode.target_reached for episode in recovery_episodes]))

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
            rendered_episode = asdict(episode)
            rendered_episode["return"] = rendered_episode.pop("return_")
            rendered_episodes.append(rendered_episode)
        return {
            "policy_path": self.policy_path,
            "max_steps": self.max_steps,
            "target_score": self.target_score,
            "score": self.capped_mean_score,
            "mean_return": self.mean_return,
            "capped_mean_score": self.capped_mean_score,
            "target_success_rate": self.target_success_rate,
            "recovery_success_rate": self.recovery_success_rate,
            "invalid_episode_rate": self.invalid_episode_rate,
            "mean_action_latency_ms": self.mean_action_latency_ms,
            "episodes": rendered_episodes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _resolve_policy_path(policy_path: str | Path) -> Path:
    path = Path(policy_path).resolve()
    if path.is_dir():
        path = path / "policy.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_policy_module(policy_path: Path) -> ModuleType:
    module_name = f"genesisbench_montezuma_submission_{abs(hash(policy_path))}"
    spec = importlib.util.spec_from_file_location(module_name, policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import policy from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
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


def _make_montezuma_env(*, seed: int, max_steps: int) -> Any:
    try:
        import envpool
    except ImportError as error:
        raise RuntimeError("Montezuma evaluation requires envpool==1.1.1") from error

    return envpool.make_gym(
        MONTEZUMA_ENV_ID,
        num_envs=1,
        batch_size=1,
        num_threads=1,
        seed=seed,
        max_episode_steps=max_steps,
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


def _reset_env(env: Any) -> np.ndarray:
    result = env.reset()
    observation = result[0] if isinstance(result, tuple) else result
    return _native_frame(observation)


def _native_frame(observation: Any) -> np.ndarray:
    array = np.asarray(observation)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            "Expected one native RGB image with three dimensions, "
            f"got shape {array.shape}"
        )
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return np.array(array, copy=True)


def _step_env(
    env: Any,
    action: int,
) -> tuple[np.ndarray, float, bool, bool]:
    result = env.step(np.asarray([action], dtype=np.int32))
    if len(result) == 5:
        observation, reward, terminated, truncated, _ = result
        terminated_value = bool(np.asarray(terminated).reshape(-1)[0])
        truncated_value = bool(np.asarray(truncated).reshape(-1)[0])
    else:
        observation, reward, done, _ = result
        terminated_value = bool(np.asarray(done).reshape(-1)[0])
        truncated_value = False
    return (
        _native_frame(observation),
        float(np.asarray(reward).reshape(-1)[0]),
        terminated_value,
        truncated_value,
    )


def _validate_action(action: Any) -> int:
    array = np.asarray(action)
    if array.shape != ():
        raise ValueError(f"Expected one scalar action, got shape {array.shape}")
    value = array.item()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Expected an integer action, got {value!r}")
    action_value = int(value)
    if not 0 <= action_value < MONTEZUMA_ACTION_COUNT:
        raise ValueError(
            f"Action must be in [0, {MONTEZUMA_ACTION_COUNT - 1}], got {action_value}"
        )
    return action_value


def _trusted_action(policy: Any, observation: np.ndarray) -> int:
    try:
        return _validate_action(policy.act(observation))
    except Exception as error:
        raise RuntimeError(
            f"Trusted bootstrap policy failed: {type(error).__name__}: {error}"
        ) from error


def _validate_variant(variant: MontezumaVariant) -> None:
    if variant.bootstrap_steps < 0:
        raise ValueError("bootstrap_steps must be non-negative")
    if variant.pre_policy_noops < 0:
        raise ValueError("pre_policy_noops must be non-negative")


def evaluate_montezuma_policy(
    policy_path: str | Path,
    *,
    seeds: Iterable[int],
    max_steps: int = 2600,
    variants: Iterable[MontezumaVariant] | None = None,
    bootstrap_policy_path: str | Path | None = None,
    target_score: float = 400.0,
    failure_score: float = 0.0,
) -> MontezumaEvaluation:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if target_score <= 0:
        raise ValueError("target_score must be positive")

    path = _resolve_policy_path(policy_path)
    module = _load_policy_module(path)
    configured_seeds = tuple(int(seed) for seed in seeds)
    configured_variants = tuple(variants or (MontezumaVariant(),))
    if not configured_seeds:
        raise ValueError("At least one seed is required")
    if not configured_variants:
        raise ValueError("At least one variant is required")
    for variant in configured_variants:
        _validate_variant(variant)

    needs_bootstrap = any(
        variant.bootstrap_steps > 0 for variant in configured_variants
    )
    bootstrap_module: ModuleType | None = None
    if needs_bootstrap:
        if bootstrap_policy_path is None:
            raise ValueError("bootstrap_policy_path is required for recovery variants")
        bootstrap_module = _load_policy_module(
            _resolve_policy_path(bootstrap_policy_path)
        )

    episodes: list[MontezumaEpisode] = []
    for variant in configured_variants:
        for seed in configured_seeds:
            env = _make_montezuma_env(seed=seed, max_steps=max_steps)
            episode_return = 0.0
            length = 0
            policy_steps = 0
            terminated = False
            truncated = False
            invalid_action = False
            policy_error: str | None = None
            latencies: list[float] = []

            try:
                observation = _reset_env(env)

                if variant.bootstrap_steps > 0:
                    assert bootstrap_module is not None
                    bootstrap_policy = _instantiate_policy(
                        bootstrap_module,
                        seed,
                    )
                    _reset_policy(bootstrap_policy, seed)
                    for _ in range(variant.bootstrap_steps):
                        if length >= max_steps or terminated or truncated:
                            break
                        action = _trusted_action(
                            bootstrap_policy,
                            observation,
                        )
                        (
                            observation,
                            reward,
                            terminated,
                            truncated,
                        ) = _step_env(env, action)
                        episode_return += reward
                        length += 1

                for _ in range(variant.pre_policy_noops):
                    if length >= max_steps or terminated or truncated:
                        break
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                    ) = _step_env(env, 0)
                    episode_return += reward
                    length += 1

                policy = _instantiate_policy(module, seed)
                _reset_policy(policy, seed)
                while length < max_steps and not (terminated or truncated):
                    started_at = time.perf_counter()
                    try:
                        action = _validate_action(policy.act(observation))
                    except Exception as error:
                        policy_error = f"{type(error).__name__}: {error}"
                        invalid_action = isinstance(error, ValueError)
                        episode_return = failure_score
                        break
                    latencies.append((time.perf_counter() - started_at) * 1000.0)
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                    ) = _step_env(env, action)
                    episode_return += reward
                    length += 1
                    policy_steps += 1

                if length >= max_steps and not terminated and policy_error is None:
                    truncated = True
            finally:
                env.close()

            episodes.append(
                MontezumaEpisode(
                    seed=seed,
                    variant=variant.name,
                    bootstrap_steps=variant.bootstrap_steps,
                    pre_policy_noops=variant.pre_policy_noops,
                    return_=float(episode_return),
                    length=length,
                    policy_steps=policy_steps,
                    target_reached=episode_return >= target_score,
                    terminated=terminated,
                    truncated=truncated,
                    invalid_action=invalid_action,
                    policy_error=policy_error,
                    mean_action_latency_ms=float(np.mean(latencies))
                    if latencies
                    else 0.0,
                )
            )

    return MontezumaEvaluation(
        policy_path=str(path),
        max_steps=max_steps,
        target_score=target_score,
        episodes=tuple(episodes),
    )
