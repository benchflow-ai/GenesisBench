from __future__ import annotations

import inspect
import json
import math
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import numpy as np

from genesisbench.policy_isolation import (
    close_policy,
    instantiate_policy,
    load_policy_module,
)


@dataclass(frozen=True)
class DynamicsVariant:
    name: str = "nominal"
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    damping_scale: float = 1.0
    actuator_scale: float = 1.0


@dataclass(frozen=True)
class HalfCheetahEpisode:
    seed: int
    variant: str
    return_: float
    length: int
    x_position: float
    x_velocity: float
    forward_reward: float
    control_cost: float
    terminated: bool
    truncated: bool
    invalid_action: bool
    policy_error: str | None
    mean_action_latency_ms: float


@dataclass(frozen=True)
class HalfCheetahEvaluation:
    policy_path: str
    max_steps: int
    episodes: tuple[HalfCheetahEpisode, ...]

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
    def mean_x_position(self) -> float:
        return float(np.mean([episode.x_position for episode in self.episodes]))

    @property
    def mean_forward_reward(self) -> float:
        return float(np.mean([episode.forward_reward for episode in self.episodes]))

    @property
    def mean_control_cost(self) -> float:
        return float(np.mean([episode.control_cost for episode in self.episodes]))

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
        rendered_episodes: list[dict[str, Any]] = []
        for episode in self.episodes:
            rendered = asdict(episode)
            rendered["return"] = rendered.pop("return_")
            rendered_episodes.append(rendered)
        return {
            "policy_path": self.policy_path,
            "max_steps": self.max_steps,
            "score": self.mean_return,
            "mean_return": self.mean_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "mean_x_position": self.mean_x_position,
            "mean_forward_reward": self.mean_forward_reward,
            "mean_control_cost": self.mean_control_cost,
            "invalid_episode_rate": self.invalid_episode_rate,
            "mean_action_latency_ms": self.mean_action_latency_ms,
            "episodes": rendered_episodes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _load_policy_module(policy_path: Path) -> ModuleType:
    module_name = (
        f"genesisbench_halfcheetah_submission_{abs(hash(policy_path.resolve()))}"
    )
    return load_policy_module(
        policy_path,
        module_name=module_name,
        add_parent_to_path=True,
    )


def _instantiate_policy(module: ModuleType, seed: int) -> Any:
    return instantiate_policy(module, init_kwargs={"seed": seed})


def _configure_policy(
    policy: Any,
    *,
    model_xml_path: Path,
    frame_skip: int,
) -> None:
    configure = getattr(policy, "configure_simulator", None)
    if configure is None:
        return

    available = {
        "model_xml_path": str(model_xml_path),
        "frame_skip": int(frame_skip),
    }
    try:
        signature = inspect.signature(configure)
    except (TypeError, ValueError):
        configure(**available)
        return

    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = (
        available
        if accepts_kwargs
        else {
            name: value
            for name, value in available.items()
            if name in signature.parameters
        }
    )
    configure(**kwargs)


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


def _halfcheetah_xml_path() -> Path:
    import gymnasium.envs.mujoco

    return (
        Path(gymnasium.envs.mujoco.__file__).resolve().parent
        / "assets"
        / "half_cheetah.xml"
    )


def _scale_vector(value: str, scale: float) -> str:
    return " ".join(f"{float(item) * scale:.12g}" for item in value.split())


def _write_variant_xml(
    variant: DynamicsVariant,
    output_path: Path,
) -> None:
    root = ET.parse(_halfcheetah_xml_path()).getroot()

    compiler = root.find("compiler")
    if compiler is not None and "settotalmass" in compiler.attrib:
        compiler.set(
            "settotalmass",
            f"{float(compiler.attrib['settotalmass']) * variant.mass_scale:.12g}",
        )

    for geom in root.findall(".//geom"):
        if "friction" in geom.attrib:
            geom.set(
                "friction",
                _scale_vector(geom.attrib["friction"], variant.friction_scale),
            )

    for joint in root.findall(".//joint"):
        if "damping" in joint.attrib:
            joint.set(
                "damping",
                f"{float(joint.attrib['damping']) * variant.damping_scale:.12g}",
            )

    for motor in root.findall(".//motor"):
        if "gear" in motor.attrib:
            motor.set(
                "gear",
                _scale_vector(motor.attrib["gear"], variant.actuator_scale),
            )

    ET.ElementTree(root).write(output_path, encoding="unicode")


def _is_nominal(variant: DynamicsVariant) -> bool:
    return variant.name == "nominal" and all(
        math.isclose(value, 1.0)
        for value in (
            variant.mass_scale,
            variant.friction_scale,
            variant.damping_scale,
            variant.actuator_scale,
        )
    )


def _make_halfcheetah_env(
    *,
    variant: DynamicsVariant,
    max_steps: int,
) -> tuple[gym.Env, tempfile.TemporaryDirectory[str] | None, Path]:
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    model_xml_path = _halfcheetah_xml_path()
    kwargs: dict[str, Any] = {"max_episode_steps": max_steps}

    if not _is_nominal(variant):
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="genesisbench-simulation-heuristics-halfcheetah-v1-"
        )
        model_xml_path = Path(temporary_directory.name) / f"{variant.name}.xml"
        _write_variant_xml(variant, model_xml_path)
        kwargs["xml_file"] = str(model_xml_path)

    env = gym.make("HalfCheetah-v5", **kwargs)
    return env, temporary_directory, model_xml_path


def _validate_action(action: Any) -> np.ndarray:
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (6,):
        raise ValueError(f"Expected action shape (6,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Action contains NaN or infinite values")
    return np.clip(array, -1.0, 1.0).astype(np.float32)


def evaluate_halfcheetah_policy(
    policy_path: str | Path,
    *,
    seeds: Iterable[int],
    max_steps: int = 1000,
    variants: Iterable[DynamicsVariant] | None = None,
    failure_return: float = -1000.0,
) -> HalfCheetahEvaluation:
    path = Path(policy_path).resolve()
    if path.is_dir():
        path = path / "policy.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("At least one seed is required")

    module = _load_policy_module(path)
    configured_variants = tuple(variants or (DynamicsVariant(),))
    episodes: list[HalfCheetahEpisode] = []

    for variant in configured_variants:
        for seed in seed_values:
            env, temporary_directory, model_xml_path = _make_halfcheetah_env(
                variant=variant,
                max_steps=max_steps,
            )
            observation, _ = env.reset(seed=seed)
            episode_return = 0.0
            forward_reward = 0.0
            control_cost = 0.0
            terminated = False
            truncated = False
            invalid_action = False
            policy_error: str | None = None
            latencies: list[float] = []
            info: dict[str, Any] = {}
            length = 0
            policy: Any | None = None

            try:
                try:
                    policy = _instantiate_policy(module, seed)
                    _configure_policy(
                        policy,
                        model_xml_path=model_xml_path,
                        frame_skip=int(env.unwrapped.frame_skip),
                    )
                    _reset_policy(policy, seed)
                except Exception as error:
                    policy_error = f"{type(error).__name__}: {error}"
                    episode_return = failure_return
                else:
                    for step_index in range(1, max_steps + 1):
                        length = step_index
                        started_at = time.perf_counter()
                        try:
                            action = _validate_action(policy.act(observation))
                        except Exception as error:
                            policy_error = f"{type(error).__name__}: {error}"
                            invalid_action = isinstance(error, ValueError)
                            episode_return = failure_return
                            break
                        latencies.append((time.perf_counter() - started_at) * 1000)
                        (
                            observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = env.step(action)
                        episode_return += float(reward)
                        forward_reward += float(info.get("reward_forward", 0.0))
                        control_cost += -float(info.get("reward_ctrl", 0.0))
                        if terminated or truncated:
                            break
            finally:
                if policy is not None:
                    close_policy(policy)
                x_position = float(env.unwrapped.data.qpos[0])
                x_velocity = float(env.unwrapped.data.qvel[0])
                env.close()
                if temporary_directory is not None:
                    temporary_directory.cleanup()

            episodes.append(
                HalfCheetahEpisode(
                    seed=seed,
                    variant=variant.name,
                    return_=float(episode_return),
                    length=length,
                    x_position=x_position,
                    x_velocity=x_velocity,
                    forward_reward=float(forward_reward),
                    control_cost=float(control_cost),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    invalid_action=invalid_action,
                    policy_error=policy_error,
                    mean_action_latency_ms=float(np.mean(latencies))
                    if latencies
                    else 0.0,
                )
            )

    return HalfCheetahEvaluation(
        policy_path=str(path),
        max_steps=max_steps,
        episodes=tuple(episodes),
    )
