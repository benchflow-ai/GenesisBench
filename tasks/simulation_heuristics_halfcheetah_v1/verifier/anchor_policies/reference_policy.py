# SPDX-License-Identifier: GPL-3.0-or-later
"""Staged asymmetric CPG/PD controller with two-level online tree MPC."""

from __future__ import annotations

import math

import mujoco
import mujoco.rollout
import numpy as np


BASE_GAIT = np.asarray(
    [
        2.0720429776207037,
        2.0890962589603244,
        0.7454172634828156,
        -0.06577692937619559,
        0.6454089230776165,
        -0.6746024815060896,
        -0.5255276137795621,
        -0.776911900431121,
        0.4784639889085752,
        -1.5344226994990133,
        0.8642039688512475,
        -0.6508749004237321,
        -0.8139980805865409,
        -0.43273458554686073,
        0.9311256504045836,
        -0.3884941211535119,
        0.5058403771772146,
        0.29329785944557646,
        -0.41630444646878295,
        -0.17793318904003722,
        0.25287195521722133,
        0.1864602367775759,
        -0.2174643453198326,
        0.32053306369231493,
        0.18857020059058224,
        0.2877837436012836,
        -0.46494131953713835,
        -0.23362679675599532,
        -0.179250145837136,
        -0.2538333727428432,
        -0.19506084586961667,
        -0.3065628153040692,
    ],
    dtype=np.float64,
)
PHASE_SUBSTEPS = 10
PLANNING_HORIZON = 14
TERMINAL_VELOCITY_WEIGHT = 0.5
PD_KP = 8.0
PD_KD = 0.0
TREE_WIDTH = 8
FAST_STAGE_STEP = 300
FINAL_STAGE_STEP = 900
COORDINATE_DELTAS = (
    -0.10,
    0.10,
    -0.25,
    0.25,
    -0.50,
    0.50,
    -0.75,
    0.75,
    -1.00,
    1.00,
)
RANDOM_BLOCKS = ((0.15, 64), (0.35, 128), (0.70, 192))
BANG_BANG_ACTIONS = np.asarray(
    np.meshgrid(*[[-1.0, 1.0]] * 6), dtype=np.float64
).T.reshape(-1, 6)


def _frequency(raw_value: np.ndarray) -> np.ndarray:
    sigmoid = 1.0 / (1.0 + np.exp(-raw_value))
    return 0.5 + 4.5 * sigmoid


def _stage_gait(amplitude: float, front_lower_leg_bias: float) -> np.ndarray:
    gait = BASE_GAIT.copy()
    gait[0] -= 1.80
    gait[1] -= 1.10
    coefficients = gait[2:].reshape(5, 6).copy()
    coefficients[1:] *= amplitude
    coefficients[0, 4:] += front_lower_leg_bias
    gait[2:] = coefficients.reshape(-1)
    return gait


START_GAIT = _stage_gait(1.15, 0.15)
FAST_GAIT = _stage_gait(1.18, 0.20)
FINAL_GAIT = START_GAIT.copy()


def _advance_phase(
    gait: np.ndarray,
    phase: np.ndarray,
    dt: float,
) -> np.ndarray:
    stance_frequency = _frequency(gait[:, 0])
    swing_frequency = _frequency(gait[:, 1])
    updated = phase.copy()
    for _ in range(PHASE_SUBSTEPS):
        frequency = np.where(
            np.sin(updated) > 0.0,
            swing_frequency,
            stance_frequency,
        )
        updated = (updated + 2.0 * math.pi * frequency * (dt / PHASE_SUBSTEPS)) % (
            2.0 * math.pi
        )
    return updated


def _cpg_actions(
    gait: np.ndarray,
    observation: np.ndarray,
    phase: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    next_phase = _advance_phase(gait, phase, dt)
    features = np.stack(
        (
            np.ones_like(next_phase),
            np.sin(next_phase),
            np.cos(next_phase),
            np.sin(2.0 * next_phase),
            np.cos(2.0 * next_phase),
        ),
        axis=1,
    )
    target = np.einsum(
        "nf,nfa->na",
        features,
        gait[:, 2:].reshape(-1, 5, 6),
    )
    action = PD_KP * (target - observation[:, 2:8]) - PD_KD * observation[:, 11:17]
    return np.clip(action, -1.0, 1.0).astype(np.float32), next_phase


def _single_cpg_action(
    gait: np.ndarray,
    observation: np.ndarray,
    phase: float,
    dt: float,
) -> tuple[np.ndarray, float]:
    next_phase = _advance_phase(
        gait[None, :],
        np.asarray([phase], dtype=np.float64),
        dt,
    )
    phase_value = float(next_phase[0])
    features = np.asarray(
        [
            1.0,
            math.sin(phase_value),
            math.cos(phase_value),
            math.sin(2.0 * phase_value),
            math.cos(2.0 * phase_value),
        ],
        dtype=np.float64,
    )
    target = features @ gait[2:].reshape(5, 6)
    action = PD_KP * (target - observation[2:8]) - PD_KD * observation[11:17]
    return action, phase_value


def _model_state(data: mujoco.MjData) -> np.ndarray:
    return np.concatenate(([data.time], data.qpos.copy(), data.qvel.copy())).astype(
        np.float64
    )


def _observations_from_states(
    model: mujoco.MjModel,
    state: np.ndarray,
) -> np.ndarray:
    qpos_start = 1
    qvel_start = qpos_start + model.nq
    qpos = state[:, qpos_start:qvel_start]
    qvel = state[:, qvel_start : qvel_start + model.nv]
    return np.concatenate((qpos[:, 1:], qvel), axis=1)


def _step_batch(
    model: mujoco.MjModel,
    rollout_data: mujoco.MjData,
    state: np.ndarray,
    action: np.ndarray,
    frame_skip: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    controls = np.repeat(
        action[:, None, :],
        frame_skip,
        axis=1,
    ).astype(np.float64)
    states, _ = mujoco.rollout.rollout(
        model,
        rollout_data,
        initial_state=state,
        control=controls,
    )
    next_state = states[:, -1, :]
    forward_reward = (next_state[:, 1] - state[:, 1]) / dt
    control_cost = 0.1 * np.sum(action * action, axis=1)
    return next_state, forward_reward - control_cost


def _score_actions(
    *,
    gait: np.ndarray,
    model: mujoco.MjModel,
    rollout_data: mujoco.MjData,
    initial_state: np.ndarray,
    phase: np.ndarray,
    first_actions: np.ndarray,
    frame_skip: int,
    dt: float,
) -> np.ndarray:
    count = len(first_actions)
    repeated_gait = np.repeat(gait[None, :], count, axis=0)
    simulated_phase = _advance_phase(repeated_gait, phase, dt)
    state, score = _step_batch(
        model,
        rollout_data,
        initial_state,
        np.clip(first_actions, -1.0, 1.0),
        frame_skip,
        dt,
    )

    for _ in range(1, PLANNING_HORIZON):
        action, simulated_phase = _cpg_actions(
            repeated_gait,
            _observations_from_states(model, state),
            simulated_phase,
            dt,
        )
        state, reward = _step_batch(
            model,
            rollout_data,
            state,
            action,
            frame_skip,
            dt,
        )
        score += reward

    qvel_start = 1 + model.nq
    return score + TERMINAL_VELOCITY_WEIGHT * state[:, qvel_start]


def _candidate_actions(
    base_action: np.ndarray,
    rng: np.random.Generator,
    *,
    include_random: bool,
) -> np.ndarray:
    candidates = [base_action]
    for joint in range(6):
        for delta in COORDINATE_DELTAS:
            candidate = base_action.copy()
            candidate[joint] = np.clip(
                candidate[joint] + delta,
                -1.0,
                1.0,
            )
            candidates.append(candidate)

    if include_random:
        for standard_deviation, count in RANDOM_BLOCKS:
            samples = base_action + standard_deviation * rng.standard_normal((count, 6))
            candidates.extend(np.clip(samples, -1.0, 1.0))

    candidates.extend(BANG_BANG_ACTIONS)
    return np.asarray(candidates, dtype=np.float64)


class Policy:
    def __init__(self, seed: int = 0) -> None:
        self._model: mujoco.MjModel | None = None
        self._state_data: mujoco.MjData | None = None
        self._rollout_data: mujoco.MjData | None = None
        self._frame_skip = 5
        self._dt = 0.05
        self.reset(seed)

    def configure_simulator(
        self,
        *,
        model_xml_path: str,
        frame_skip: int = 5,
    ) -> None:
        model = mujoco.MjModel.from_xml_path(model_xml_path)
        if (model.nq, model.nv, model.nu) != (9, 9, 6):
            raise ValueError(
                "Expected HalfCheetah model dimensions (9, 9, 6), "
                f"got {(model.nq, model.nv, model.nu)}"
            )
        self._model = model
        self._state_data = mujoco.MjData(model)
        self._rollout_data = mujoco.MjData(model)
        self._frame_skip = int(frame_skip)
        self._dt = float(model.opt.timestep) * self._frame_skip

    def reset(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.phase = 0.0
        self.step = 0
        self.rng = np.random.default_rng(self.seed + 333)

    def _gait(self) -> np.ndarray:
        if self.step < FAST_STAGE_STEP:
            return START_GAIT
        if self.step < FINAL_STAGE_STEP:
            return FAST_GAIT
        return FINAL_GAIT

    def _sync_observation(self, observation: np.ndarray) -> None:
        if self._model is None or self._state_data is None:
            raise RuntimeError("configure_simulator must be called before act")
        if observation.shape != (17,):
            raise ValueError(
                f"Expected observation shape (17,), got {observation.shape}"
            )
        self._state_data.time = self.step * self._dt
        self._state_data.qpos[0] = 0.0
        self._state_data.qpos[1:] = observation[:8]
        self._state_data.qvel[:] = observation[8:]
        self._state_data.ctrl[:] = 0.0
        mujoco.mj_forward(self._model, self._state_data)

    def act(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        self._sync_observation(observation)
        assert self._model is not None
        assert self._state_data is not None
        assert self._rollout_data is not None

        gait = self._gait()
        base_action, next_phase = _single_cpg_action(
            gait,
            observation,
            self.phase,
            self._dt,
        )
        first_candidates = _candidate_actions(
            np.clip(base_action, -1.0, 1.0),
            self.rng,
            include_random=True,
        )
        initial_state = np.repeat(
            _model_state(self._state_data)[None, :],
            len(first_candidates),
            axis=0,
        )
        first_scores = _score_actions(
            gait=gait,
            model=self._model,
            rollout_data=self._rollout_data,
            initial_state=initial_state,
            phase=np.full(
                len(first_candidates),
                self.phase,
                dtype=np.float64,
            ),
            first_actions=first_candidates,
            frame_skip=self._frame_skip,
            dt=self._dt,
        )
        top_indices = np.argsort(first_scores)[-TREE_WIDTH:]
        top_first_actions = first_candidates[top_indices]

        top_count = len(top_first_actions)
        top_initial_state = np.repeat(
            _model_state(self._state_data)[None, :],
            top_count,
            axis=0,
        )
        second_state, first_reward = _step_batch(
            self._model,
            self._rollout_data,
            top_initial_state,
            top_first_actions,
            self._frame_skip,
            self._dt,
        )
        second_phase = np.full(
            top_count,
            next_phase,
            dtype=np.float64,
        )
        repeated_gait = np.repeat(gait[None, :], top_count, axis=0)
        second_base_actions, _ = _cpg_actions(
            repeated_gait,
            _observations_from_states(self._model, second_state),
            second_phase.copy(),
            self._dt,
        )

        second_action_sets = [
            _candidate_actions(
                second_base_actions[branch].astype(np.float64),
                self.rng,
                include_random=False,
            )
            for branch in range(top_count)
        ]
        branch_indices = np.concatenate(
            [
                np.full(len(actions), branch, dtype=np.int64)
                for branch, actions in enumerate(second_action_sets)
            ]
        )
        second_actions = np.concatenate(second_action_sets, axis=0)
        second_initial_state = np.concatenate(
            [
                np.repeat(
                    second_state[branch : branch + 1],
                    len(actions),
                    axis=0,
                )
                for branch, actions in enumerate(second_action_sets)
            ],
            axis=0,
        )
        second_initial_phase = np.concatenate(
            [
                np.full(
                    len(actions),
                    second_phase[branch],
                    dtype=np.float64,
                )
                for branch, actions in enumerate(second_action_sets)
            ]
        )
        tree_scores = first_reward[branch_indices] + _score_actions(
            gait=gait,
            model=self._model,
            rollout_data=self._rollout_data,
            initial_state=second_initial_state,
            phase=second_initial_phase,
            first_actions=second_actions,
            frame_skip=self._frame_skip,
            dt=self._dt,
        )

        best_branch = branch_indices[int(np.argmax(tree_scores))]
        action = top_first_actions[best_branch]
        self.phase = next_phase
        self.step += 1
        return np.clip(action, -1.0, 1.0)
