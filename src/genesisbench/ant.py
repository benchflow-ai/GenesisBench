from __future__ import annotations

import importlib.util
import inspect
import json
import math
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class DynamicsVariant:
    name: str = "nominal"
    density_scale: float = 1.0
    friction_scale: float = 1.0
    damping_scale: float = 1.0
    actuator_scale: float = 1.0


@dataclass(frozen=True)
class AntEpisode:
    seed: int
    variant: str
    return_: float
    length: int
    x_position: float
    x_velocity: float
    terminated: bool
    truncated: bool
    invalid_action: bool
    policy_error: str | None
    mean_action_latency_ms: float


@dataclass(frozen=True)
class AntEvaluation:
    policy_path: str
    max_steps: int
    episodes: tuple[AntEpisode, ...]

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
    def fall_rate(self) -> float:
        return float(
            np.mean(
                [
                    episode.terminated and episode.length < self.max_steps
                    for episode in self.episodes
                ]
            )
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
            "score": self.mean_return,
            "mean_return": self.mean_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "mean_x_position": self.mean_x_position,
            "fall_rate": self.fall_rate,
            "invalid_episode_rate": self.invalid_episode_rate,
            "mean_action_latency_ms": self.mean_action_latency_ms,
            "episodes": rendered_episodes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _load_policy_module(policy_path: Path) -> ModuleType:
    module_name = f"genesisbench_submission_{abs(hash(policy_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        policy_path,
    )
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


def _ant_xml_path() -> Path:
    import gymnasium.envs.mujoco

    return Path(gymnasium.envs.mujoco.__file__).resolve().parent / "assets" / "ant.xml"


def _scale_vector(value: str, scale: float) -> str:
    return " ".join(f"{float(item) * scale:.12g}" for item in value.split())


def _write_variant_xml(variant: DynamicsVariant, output_path: Path) -> None:
    root = ET.parse(_ant_xml_path()).getroot()

    for default in root.findall("./default"):
        geom = default.find("geom")
        if geom is not None:
            if "density" in geom.attrib:
                geom.set(
                    "density",
                    f"{float(geom.attrib['density']) * variant.density_scale:.12g}",
                )
            if "friction" in geom.attrib:
                geom.set(
                    "friction",
                    _scale_vector(geom.attrib["friction"], variant.friction_scale),
                )
        joint = default.find("joint")
        if joint is not None and "damping" in joint.attrib:
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


def _make_ant_env(
    *,
    variant: DynamicsVariant,
    max_steps: int,
) -> tuple[
    gym.Env,
    tempfile.TemporaryDirectory[str] | None,
    Path,
]:
    temporary_directory = tempfile.TemporaryDirectory(
        prefix="genesisbench-simulation-heuristics-ant-v1-"
    )
    model_xml_path = Path(temporary_directory.name) / f"{variant.name}.xml"
    kwargs: dict[str, Any] = {
        "include_cfrc_ext_in_observation": False,
        "contact_cost_weight": 0.0,
        "max_episode_steps": max_steps,
    }
    is_variant = variant.name != "nominal" or any(
        not math.isclose(value, 1.0)
        for value in (
            variant.density_scale,
            variant.friction_scale,
            variant.damping_scale,
            variant.actuator_scale,
        )
    )
    if is_variant:
        _write_variant_xml(variant, model_xml_path)
    else:
        shutil.copy2(_ant_xml_path(), model_xml_path)
    kwargs["xml_file"] = str(model_xml_path)

    return (
        gym.make("Ant-v5", **kwargs),
        temporary_directory,
        model_xml_path,
    )


def _validate_action(action: Any) -> np.ndarray:
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (8,):
        raise ValueError(f"Expected action shape (8,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Action contains NaN or infinite values")
    return np.clip(array, -1.0, 1.0)


def evaluate_ant_policy(
    policy_path: str | Path,
    *,
    seeds: Iterable[int],
    max_steps: int = 1000,
    variants: Iterable[DynamicsVariant] | None = None,
    failure_return: float = -1000.0,
) -> AntEvaluation:
    path = Path(policy_path).resolve()
    if path.is_dir():
        path = path / "policy.py"
    if not path.is_file():
        raise FileNotFoundError(path)

    module = _load_policy_module(path)
    configured_variants = tuple(variants or (DynamicsVariant(),))
    configured_seeds = tuple(int(seed) for seed in seeds)
    episodes: list[AntEpisode] = []

    for variant in configured_variants:
        for seed in configured_seeds:
            env, temporary_directory, model_xml_path = _make_ant_env(
                variant=variant,
                max_steps=max_steps,
            )
            observation, _ = env.reset(seed=seed)
            policy = _instantiate_policy(module, seed)
            _configure_policy(
                policy,
                model_xml_path=model_xml_path,
                frame_skip=int(env.unwrapped.frame_skip),
            )
            _reset_policy(policy, seed)
            episode_return = 0.0
            terminated = False
            truncated = False
            invalid_action = False
            policy_error: str | None = None
            latencies: list[float] = []
            info: dict[str, Any] = {}
            length = 0

            try:
                for length in range(1, max_steps + 1):
                    started_at = time.perf_counter()
                    try:
                        action = _validate_action(policy.act(observation))
                    except Exception as error:
                        policy_error = f"{type(error).__name__}: {error}"
                        invalid_action = isinstance(error, ValueError)
                        episode_return = failure_return
                        break
                    latencies.append((time.perf_counter() - started_at) * 1000)
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    if terminated or truncated:
                        break
            finally:
                env.close()
                if temporary_directory is not None:
                    temporary_directory.cleanup()

            episodes.append(
                AntEpisode(
                    seed=int(seed),
                    variant=variant.name,
                    return_=float(episode_return),
                    length=length,
                    x_position=float(info.get("x_position", 0.0)),
                    x_velocity=float(info.get("x_velocity", 0.0)),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    invalid_action=invalid_action,
                    policy_error=policy_error,
                    mean_action_latency_ms=float(np.mean(latencies))
                    if latencies
                    else 0.0,
                )
            )

    return AntEvaluation(
        policy_path=str(path),
        max_steps=max_steps,
        episodes=tuple(episodes),
    )
