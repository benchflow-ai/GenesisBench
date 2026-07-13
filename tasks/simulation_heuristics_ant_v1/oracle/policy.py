# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Warm-started speed-adaptive residual MPC reference for Ant-v5.

Adapted for the GenesisBench policy API from ``mujoco/ant/heuristic_ant.py``
at learning-beyond-gradients revision
``3555c2956c257d49a5015b782cbe485b14fd659e``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


Q_INDEX = np.asarray([6, 7, 0, 1, 2, 3, 4, 5], dtype=np.int64)
LEG_PHASE = np.asarray([0.0, math.pi, 0.0, math.pi], dtype=np.float64)
HIP_SIGN = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
ANKLE_SIGN = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
HEADING_AXIS = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
PITCH_AXIS = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
ROLL_AXIS = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
FOOT_BODY_IDS = np.asarray([13, 4, 7, 10], dtype=np.int64)
FOOT_OBS_ROWS = FOOT_BODY_IDS - 1


@dataclass(frozen=True)
class ControllerConfig:
    dphi: float = 0.660934259732249
    dphi_speed_gain: float = -0.02
    dphi_speed_target: float = 5.8
    dphi_min: float = 0.62
    dphi_max: float = 0.72
    hip_bias: float = 0.12217781430672398
    hip_amp: float = 0.5705418365199333
    ankle_bias: float = 0.36651046903486795
    ankle_amp: float = 0.26587749767314783
    kp: float = 0.8108143632989734
    kd: float = 0.0
    pitch_gain: float = -0.19444590796726124
    pitch_rate_gain: float = 0.04099276700871415
    roll_gain: float = -0.25536960225655303
    roll_rate_gain: float = 0.023293075237761272
    contact_hip_gain: float = 0.0
    contact_ankle_gain: float = 0.0
    contact_push_hip_amp: float = 0.0
    contact_push_ankle_amp: float = 0.0
    contact_push_phase: float = 2.356194490192345
    contact_push_width: float = 0.55
    stance_duty: float = 0.6355364206196007
    stance_duty_speed_gain: float = -0.01
    stance_duty_speed_target: float = 5.8
    stance_duty_min: float = 0.6
    stance_duty_max: float = 0.67
    hip_stance_scale: float = 1.0479076970107701
    hip_swing_scale: float = 1.0031777685985328
    ankle_stance_scale: float = 0.976603459922793
    ankle_swing_scale: float = 0.9374473230114526
    yaw_gain: float = -0.12067720879887742
    yaw_rate_gain: float = 0.04418873596679619
    hip_h2_amp: float = 0.10975404801587477
    hip_h2_phase: float = 2.0862256065597453
    ankle_h2_amp: float = -0.003434817287963554
    ankle_h2_phase: float = 1.2927488104774438
    hip_h3_amp: float = 0.04827596673280693
    hip_h3_phase: float = -0.49944083263433436
    ankle_h3_amp: float = -0.06968988354403895
    ankle_h3_phase: float = 1.5873441034476188
    mpc_horizon: int = 10
    mpc_candidates: int = 96
    mpc_sigma: float = 0.07614211639071694
    mpc_clip: float = 0.12016284361036686
    mpc_pose_cost: float = 23.348190567885954
    mpc_pitch_target: float = 0.0
    mpc_yaw_cost: float = 2.7292168081366723
    mpc_z_cost: float = 2.1215830559511737
    mpc_z_target: float = 0.4519975076600261
    mpc_forward_weight: float = 1.0
    mpc_ctrl_cost: float = 0.5
    mpc_terminal_vel_cost: float = 0.01
    mpc_plan_decay: float = 0.504186948858276
    mpc_seed: int = 12


CONFIG = ControllerConfig()


def _warp_leg_phase(leg_phase: np.ndarray, stance_duty: float) -> np.ndarray:
    clipped_duty = float(np.clip(stance_duty, 0.05, 0.95))
    if abs(clipped_duty - 0.5) < 1e-12:
        return np.mod(leg_phase, 2.0 * math.pi)

    phase_unit = np.mod(leg_phase, 2.0 * math.pi) / (2.0 * math.pi)
    stance_unit = 0.5 * phase_unit / clipped_duty
    swing_unit = 0.5 + 0.5 * (phase_unit - clipped_duty) / (1.0 - clipped_duty)
    warped_unit = np.where(
        phase_unit < clipped_duty,
        stance_unit,
        swing_unit,
    )
    return 2.0 * math.pi * warped_unit


def _adaptive_dphi(config: ControllerConfig, x_velocity: float) -> float:
    dphi = config.dphi + config.dphi_speed_gain * (
        x_velocity - config.dphi_speed_target
    )
    return float(np.clip(dphi, config.dphi_min, config.dphi_max))


def _adaptive_stance_duty(
    config: ControllerConfig,
    x_velocity: float,
) -> float:
    stance_duty = config.stance_duty + config.stance_duty_speed_gain * (
        x_velocity - config.stance_duty_speed_target
    )
    return float(
        np.clip(
            stance_duty,
            config.stance_duty_min,
            config.stance_duty_max,
        )
    )


def _rhythmic_action(
    config: ControllerConfig,
    phase: float,
    stance_duty: float,
    q: np.ndarray,
    dq: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
    roll_rate: float,
    pitch_rate: float,
    yaw_rate: float,
    foot_contacts: np.ndarray,
) -> np.ndarray:
    leg_phase = _warp_leg_phase(phase + LEG_PHASE, stance_duty)
    stance_mask = leg_phase < math.pi
    hip_wave = config.hip_bias + np.where(
        stance_mask,
        config.hip_stance_scale,
        config.hip_swing_scale,
    ) * (
        config.hip_amp * np.sin(leg_phase)
        + config.hip_h2_amp * np.sin(2.0 * leg_phase + config.hip_h2_phase)
        + config.hip_h3_amp * np.sin(3.0 * leg_phase + config.hip_h3_phase)
    )
    ankle_wave = config.ankle_bias + np.where(
        stance_mask,
        config.ankle_stance_scale,
        config.ankle_swing_scale,
    ) * (
        config.ankle_amp * np.cos(leg_phase)
        + config.ankle_h2_amp * np.cos(2.0 * leg_phase + config.ankle_h2_phase)
        + config.ankle_h3_amp * np.cos(3.0 * leg_phase + config.ankle_h3_phase)
    )
    push_width = max(config.contact_push_width, 1e-3)
    contact_push = (
        np.exp(-0.5 * np.square((leg_phase - config.contact_push_phase) / push_width))
        * stance_mask
        * foot_contacts
    )
    hip_wave = hip_wave + config.contact_push_hip_amp * contact_push
    ankle_wave = ankle_wave + config.contact_push_ankle_amp * contact_push
    balance_wave = PITCH_AXIS * (
        config.pitch_gain * pitch + config.pitch_rate_gain * pitch_rate
    ) - ROLL_AXIS * (config.roll_gain * roll + config.roll_rate_gain * roll_rate)

    action = np.empty(8, dtype=np.float64)
    action[0::2] = (
        config.kp
        * (
            HIP_SIGN * hip_wave
            + HEADING_AXIS
            * (
                config.yaw_gain * yaw
                + config.yaw_rate_gain * yaw_rate
                + config.contact_hip_gain * foot_contacts
            )
            - q[0::2]
        )
        - config.kd * dq[0::2]
    )
    action[1::2] = (
        config.kp
        * (
            ANKLE_SIGN
            * (ankle_wave + balance_wave + config.contact_ankle_gain * foot_contacts)
            - q[1::2]
        )
        - config.kd * dq[1::2]
    )
    return np.clip(action, -1.0, 1.0)


def _euler(quaternion: np.ndarray) -> tuple[float, float, float]:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion) + 1e-12
    w, x, y, z = quaternion
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return roll, pitch, yaw


def _observation_array(observation: np.ndarray) -> np.ndarray:
    array = np.asarray(observation)
    if array.ndim == 2:
        array = array[0]
    if array.ndim != 1 or array.shape[0] < 27:
        raise ValueError(f"Unsupported Ant observation shape: {array.shape}")
    return array


def _joint_state(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = _observation_array(observation)
    return array[5:13], array[19:27]


def _forward_velocity(observation: np.ndarray) -> float:
    return float(_observation_array(observation)[13])


def _torso_state(
    observation: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    array = _observation_array(observation)
    quaternion = np.asarray(array[1:5], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion) + 1e-12
    roll, pitch, yaw = _euler(quaternion)
    return (
        roll,
        pitch,
        yaw,
        float(array[16]),
        float(array[17]),
        float(array[18]),
    )


def _observation_foot_contacts(observation: np.ndarray) -> np.ndarray:
    array = _observation_array(observation)
    if array.shape[0] < 27 + 13 * 6:
        return np.zeros(4, dtype=np.float64)
    contact_forces = array[27 : 27 + 13 * 6].reshape(13, 6)
    return np.clip(contact_forces[FOOT_OBS_ROWS, 5], 0.0, 1.0)


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self._model: mujoco.MjModel | None = None
        self._rollout_data: mujoco.MjData | None = None
        self._frame_skip = 5
        self._residual_plan = np.zeros(
            (CONFIG.mpc_horizon, 8),
            dtype=np.float64,
        )
        self.reset(seed)

    def configure_simulator(
        self,
        *,
        model_xml_path: str,
        frame_skip: int = 5,
    ) -> None:
        model = mujoco.MjModel.from_xml_path(model_xml_path)
        if (model.nq, model.nv, model.nu) != (15, 14, 8):
            raise ValueError(
                "Expected Ant model dimensions (15, 14, 8), "
                f"got {(model.nq, model.nv, model.nu)}"
            )
        self._model = model
        self._rollout_data = mujoco.MjData(model)
        self._frame_skip = int(frame_skip)

    def reset(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.phase = 0.0
        self._rng = np.random.default_rng(CONFIG.mpc_seed)
        self._residual_plan.fill(0.0)

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self._model is None or self._rollout_data is None:
            raise RuntimeError("configure_simulator must be called before act")
        observation = _observation_array(observation)
        self._set_model_state(observation)
        q = self._rollout_data.qpos[7:15][Q_INDEX]
        dq = self._rollout_data.qvel[6:14][Q_INDEX]
        x_velocity = _forward_velocity(observation)
        (
            roll,
            pitch,
            yaw,
            roll_rate,
            pitch_rate,
            yaw_rate,
        ) = _torso_state(observation)
        foot_contacts = _observation_foot_contacts(observation)
        stance_duty = _adaptive_stance_duty(CONFIG, x_velocity)
        base_action = _rhythmic_action(
            CONFIG,
            self.phase,
            stance_duty,
            q,
            dq,
            roll,
            pitch,
            yaw,
            roll_rate,
            pitch_rate,
            yaw_rate,
            foot_contacts,
        )

        best_residuals = self._residual_plan.copy()
        best_objective = self._rollout_objective(
            observation,
            best_residuals,
        )
        for _ in range(CONFIG.mpc_candidates - 1):
            residuals = np.clip(
                best_residuals
                + self._rng.normal(
                    0.0,
                    CONFIG.mpc_sigma,
                    size=(CONFIG.mpc_horizon, 8),
                ),
                -CONFIG.mpc_clip,
                CONFIG.mpc_clip,
            )
            residuals[1:] = 0.6 * residuals[1:] + 0.4 * residuals[:-1]
            objective = self._rollout_objective(observation, residuals)
            if objective > best_objective:
                best_objective = objective
                best_residuals = residuals

        self.phase += _adaptive_dphi(CONFIG, x_velocity)
        self._residual_plan[:-1] = CONFIG.mpc_plan_decay * best_residuals[1:]
        self._residual_plan[-1] = 0.0
        return np.clip(
            base_action + best_residuals[0],
            -1.0,
            1.0,
        )

    def _set_model_state(self, observation: np.ndarray) -> None:
        assert self._model is not None
        assert self._rollout_data is not None
        self._rollout_data.qpos[0:2] = 0.0
        self._rollout_data.qpos[2:] = observation[:13]
        self._rollout_data.qvel[:] = observation[13:27]
        mujoco.mj_forward(self._model, self._rollout_data)

    def _rollout_objective(
        self,
        observation: np.ndarray,
        residuals: np.ndarray,
    ) -> float:
        assert self._model is not None
        assert self._rollout_data is not None
        self._set_model_state(observation)
        objective = 0.0
        phase = self.phase
        for horizon_index, residual in enumerate(residuals):
            q = self._rollout_data.qpos[7:15][Q_INDEX]
            dq = self._rollout_data.qvel[6:14][Q_INDEX]
            x_velocity_before = float(self._rollout_data.qvel[0])
            roll, pitch, yaw = _euler(self._rollout_data.qpos[3:7])
            roll_rate = float(self._rollout_data.qvel[3])
            pitch_rate = float(self._rollout_data.qvel[4])
            yaw_rate = float(self._rollout_data.qvel[5])
            if horizon_index == 0:
                foot_contacts = _observation_foot_contacts(observation)
            else:
                foot_contacts = np.clip(
                    self._rollout_data.cfrc_ext[FOOT_BODY_IDS, 5],
                    0.0,
                    1.0,
                )
            stance_duty = _adaptive_stance_duty(
                CONFIG,
                x_velocity_before,
            )
            action = np.clip(
                _rhythmic_action(
                    CONFIG,
                    phase,
                    stance_duty,
                    q,
                    dq,
                    roll,
                    pitch,
                    yaw,
                    roll_rate,
                    pitch_rate,
                    yaw_rate,
                    foot_contacts,
                )
                + residual,
                -1.0,
                1.0,
            )
            x_before = float(self._rollout_data.qpos[0])
            self._rollout_data.ctrl[:] = action
            for _ in range(self._frame_skip):
                mujoco.mj_step(self._model, self._rollout_data)

            dt = self._frame_skip * self._model.opt.timestep
            x_velocity = (float(self._rollout_data.qpos[0]) - x_before) / dt
            z_position = float(self._rollout_data.qpos[2])
            roll, pitch, yaw = _euler(self._rollout_data.qpos[3:7])
            objective += CONFIG.mpc_forward_weight * x_velocity + (
                1.0 if 0.2 <= z_position <= 1.0 else -50.0
            )
            objective -= CONFIG.mpc_ctrl_cost * float(np.square(action).sum())
            objective -= CONFIG.mpc_pose_cost * (
                roll * roll
                + (pitch - CONFIG.mpc_pitch_target) * (pitch - CONFIG.mpc_pitch_target)
            )
            objective -= CONFIG.mpc_yaw_cost * yaw * yaw
            objective -= CONFIG.mpc_z_cost * (z_position - CONFIG.mpc_z_target) ** 2
            if z_position < 0.23 or z_position > 0.95:
                objective -= 100.0
            phase += _adaptive_dphi(CONFIG, x_velocity)

        joint_velocity = self._rollout_data.qvel[6:14]
        objective -= CONFIG.mpc_terminal_vel_cost * float(
            np.dot(joint_velocity, joint_velocity)
        )
        return objective
