"""Vectorized MDP terms for the exact-RCSS T1 full-body goalkeeper.

The actor receives only deployable robot proprioception and estimated ball state.
Episode bookkeeping and contact-force truth are reserved for rewards, metrics and
the privileged critic.  All functions operate on batched Torch tensors so the
same task can run thousands of independent MuJoCo-Warp worlds on one GPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.actions.actions import (
    JointPositionAction,
    JointPositionActionCfg,
)
from mjlab.utils.lab_api.math import quat_apply_inverse

from locomotion.t1_spec import (
    T1_JOINT_NAMES,
    T1_LOWER_LIMITS,
    T1_UPPER_LIMITS,
    clamp_t1_targets,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


ACTION_DIM = 23
ACTOR_OBS_DIM = 87
CRITIC_OBS_DIM = 107
GOAL_HALF_WIDTH_M = 1.83
GOAL_HEIGHT_M = 1.20
GOAL_LINE_X_M = -0.65
BALL_RADIUS_M = 0.11
CONTROL_HZ = 50.0

# Exact joint-level actuator-force caps in the RCSS T1 XML, in the canonical
# 23-DoF order.  These are lower than some training servo force limits and are
# therefore the effective competition caps.
T1_EFFECTIVE_TORQUE_CAPS = (
    7.0,
    7.0,
    18.0,
    18.0,
    18.0,
    18.0,
    18.0,
    18.0,
    18.0,
    18.0,
    30.0,
    45.0,
    30.0,
    30.0,
    60.0,
    20.0,
    15.0,
    45.0,
    30.0,
    30.0,
    60.0,
    20.0,
    15.0,
)


def _normalise_joint_name(name: str) -> str:
    return "Head_yaw" if name == "AAHead_yaw" else name


class T1ClampedJointPositionAction(JointPositionAction):
    """23-D action with finite checks and canonical T1 target clamping.

    ``clamp_t1_targets`` is NumPy/single-robot code.  At construction it is used
    to obtain the canonical bounds; the identical bounds are then applied with
    Torch to every GPU world in one batched operation.  Intersecting them with
    the exact XML limits also accounts for the XML's rounded radian constants.
    """

    def __init__(self, cfg: T1ClampedJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        names = tuple(_normalise_joint_name(name) for name in self.target_names)
        if self.action_dim != ACTION_DIM or names != tuple(T1_JOINT_NAMES):
            raise ValueError(
                "v047 action order must be the canonical 23-DoF T1 order; "
                f"received {self.action_dim} targets: {names}"
            )

        # Passing the canonical arrays through the public single-robot clamp
        # catches order/shape drift.  ``clamp_t1_targets`` deliberately maps
        # infinities to zero, so infinities cannot be used to query its bounds.
        canonical_lower = clamp_t1_targets(T1_LOWER_LIMITS.copy())
        canonical_upper = clamp_t1_targets(T1_UPPER_LIMITS.copy())
        xml_limits = self._entity.data.joint_pos_limits[:, self._target_ids]
        lower = torch.as_tensor(canonical_lower, device=self.device).view(1, -1)
        upper = torch.as_tensor(canonical_upper, device=self.device).view(1, -1)
        self._lower = torch.maximum(lower, xml_limits[..., 0])
        self._upper = torch.minimum(upper, xml_limits[..., 1])
        if not bool(torch.all(self._lower < self._upper)):
            raise ValueError("canonical and XML T1 joint limits do not overlap")

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(
                f"v047 action tensor must be ({self.num_envs}, {ACTION_DIM}), "
                f"received {tuple(actions.shape)}"
            )
        if not bool(torch.isfinite(actions).all()):
            raise FloatingPointError("v047 refuses non-finite policy actions")
        super().process_actions(torch.clamp(actions, -1.0, 1.0))

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = self._processed_actions - encoder_bias
        target = torch.maximum(torch.minimum(target, self._upper), self._lower)
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


class T1ClampedJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env: ManagerBasedRlEnv) -> T1ClampedJointPositionAction:
        return T1ClampedJointPositionAction(self, env)


def _ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _zeros(env: ManagerBasedRlEnv, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=dtype)


def goalkeeper_state(env: ManagerBasedRlEnv) -> SimpleNamespace:
    """Return the single shared episode-state object, creating it when needed."""

    existing = getattr(env, "_v047_goalkeeper_state", None)
    if existing is not None:
        return existing
    state = SimpleNamespace(
        curriculum_stage=_zeros(env, dtype=torch.long),
        initial_speed=_zeros(env),
        shot_forward_speed=_zeros(env),
        shot_target_y=_zeros(env),
        shot_target_z=_zeros(env),
        shot_origin_y=_zeros(env),
        disturbance_level=_zeros(env),
        ever_legal=_zeros(env, dtype=torch.bool),
        ever_foot_or_calf=_zeros(env, dtype=torch.bool),
        ever_body=_zeros(env, dtype=torch.bool),
        ever_head=_zeros(env, dtype=torch.bool),
        ever_hand=_zeros(env, dtype=torch.bool),
        current_legal=_zeros(env, dtype=torch.bool),
        current_head=_zeros(env, dtype=torch.bool),
        current_hand=_zeros(env, dtype=torch.bool),
        peak_foot_force=_zeros(env),
        peak_body_force=_zeros(env),
        peak_head_force=_zeros(env),
        peak_hand_force=_zeros(env),
        safe_steps=_zeros(env, dtype=torch.long),
        post_block_stand_steps=_zeros(env, dtype=torch.long),
        torque_saturation_steps=_zeros(env, dtype=torch.long),
        blocked=_zeros(env, dtype=torch.bool),
        new_block=_zeros(env, dtype=torch.bool),
        conceded=_zeros(env, dtype=torch.bool),
        fell=_zeros(env, dtype=torch.bool),
        new_fall=_zeros(env, dtype=torch.bool),
        joint_violation=_zeros(env, dtype=torch.bool),
        unsafe=_zeros(env, dtype=torch.bool),
        nonfinite=_zeros(env, dtype=torch.bool),
        success=_zeros(env, dtype=torch.bool),
        new_success=_zeros(env, dtype=torch.bool),
        last_transition_step=torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        ),
    )
    env._v047_goalkeeper_state = state
    return state


_BOOL_STATE_FIELDS = (
    "ever_legal",
    "ever_foot_or_calf",
    "ever_body",
    "ever_head",
    "ever_hand",
    "current_legal",
    "current_head",
    "current_hand",
    "blocked",
    "new_block",
    "conceded",
    "fell",
    "new_fall",
    "joint_violation",
    "unsafe",
    "nonfinite",
    "success",
    "new_success",
)
_FLOAT_STATE_FIELDS = (
    "initial_speed",
    "shot_forward_speed",
    "shot_target_y",
    "shot_target_z",
    "shot_origin_y",
    "disturbance_level",
    "peak_foot_force",
    "peak_body_force",
    "peak_head_force",
    "peak_hand_force",
)
_LONG_STATE_FIELDS = (
    "curriculum_stage",
    "safe_steps",
    "post_block_stand_steps",
    "torque_saturation_steps",
)


def reset_goalkeeper_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice) -> None:
    state = goalkeeper_state(env)
    ids = _ids(env, env_ids)
    for name in _BOOL_STATE_FIELDS:
        getattr(state, name)[ids] = False
    for name in _FLOAT_STATE_FIELDS:
        getattr(state, name)[ids] = 0.0
    for name in _LONG_STATE_FIELDS:
        getattr(state, name)[ids] = 0
    state.last_transition_step[ids] = -1


def _uniform(
    count: int, low: float, high: float, *, device: str
) -> torch.Tensor:
    if low == high:
        return torch.full((count,), low, device=device)
    return low + (high - low) * torch.rand(count, device=device)


def _stage_from_counter(counter: int, thresholds: tuple[int, int, int]) -> int:
    return int(counter >= thresholds[0]) + int(counter >= thresholds[1]) + int(
        counter >= thresholds[2]
    )


def reset_goalkeeper_scenario(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    stage_thresholds: tuple[int, int, int],
    stage_override: int | None = None,
    evaluation_cell: tuple[int, int, int] | None = None,
) -> None:
    """Reset a standing keeper and fire one randomized, physically simulated shot."""

    ids = _ids(env, env_ids)
    count = len(ids)
    reset_goalkeeper_state(env, ids)
    state = goalkeeper_state(env)
    stage = (
        _stage_from_counter(env.common_step_counter, stage_thresholds)
        if stage_override is None
        else int(stage_override)
    )
    if stage not in (0, 1, 2, 3):
        raise ValueError(f"v047 curriculum stage must be 0..3, received {stage}")
    state.curriculum_stage[ids] = stage

    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    origins = env.scene.env_origins[ids]

    robot_root = robot.data.default_root_state[ids].clone()
    robot_root[:, 0:3] += origins
    lateral_jitter = (0.0, 0.04, 0.08, 0.12)[stage]
    yaw_jitter = (0.0, 0.02, 0.06, 0.12)[stage]
    root_vel_jitter = (0.0, 0.0, 0.08, 0.20)[stage]
    robot_root[:, 1] += _uniform(
        count, -lateral_jitter, lateral_jitter, device=env.device
    )
    yaw = _uniform(count, -yaw_jitter, yaw_jitter, device=env.device)
    robot_root[:, 3] = torch.cos(0.5 * yaw)
    robot_root[:, 4:6] = 0.0
    robot_root[:, 6] = torch.sin(0.5 * yaw)
    robot_root[:, 7:13] = 0.0
    robot_root[:, 7:9] = torch.stack(
        (
            _uniform(count, -root_vel_jitter, root_vel_jitter, device=env.device),
            _uniform(count, -root_vel_jitter, root_vel_jitter, device=env.device),
        ),
        dim=-1,
    )
    robot.write_root_state_to_sim(robot_root, env_ids=ids)

    joint_pos = robot.data.default_joint_pos[ids].clone()
    joint_vel = robot.data.default_joint_vel[ids].clone()
    if stage >= 3:
        joint_pos += _uniform(count * ACTION_DIM, -0.015, 0.015, device=env.device).view(
            count, ACTION_DIM
        )
    hard_limits = robot.data.joint_pos_limits[ids]
    joint_pos = torch.maximum(
        torch.minimum(joint_pos, hard_limits[..., 1]), hard_limits[..., 0]
    )
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)

    spawn_x_ranges = ((4.8, 5.2), (4.5, 5.8), (4.2, 6.0), (4.0, 6.3))
    speed_ranges = ((3.5, 4.5), (3.5, 6.0), (4.0, 8.0), (3.0, 9.0))
    target_y_ranges = ((-0.45, 0.45), (-1.20, 1.20), (-1.72, 1.72), (-1.78, 1.78))
    target_z_ranges = ((BALL_RADIUS_M, BALL_RADIUS_M), (BALL_RADIUS_M, 0.20), (BALL_RADIUS_M, 0.90), (BALL_RADIUS_M, 1.12))
    spawn_y_ranges = ((-0.45, 0.45), (-0.85, 0.85), (-1.40, 1.40), (-1.80, 1.80))

    spawn_x = _uniform(count, *spawn_x_ranges[stage], device=env.device)
    if evaluation_cell is None:
        target_y = _uniform(count, *target_y_ranges[stage], device=env.device)
        target_z = _uniform(count, *target_z_ranges[stage], device=env.device)
        forward_speed = _uniform(count, *speed_ranges[stage], device=env.device)
    else:
        lateral_cell, speed_cell, height_cell = (int(v) for v in evaluation_cell)
        if not all(0 <= value < 3 for value in (lateral_cell, speed_cell, height_cell)):
            raise ValueError(f"invalid v047 evaluation cell: {evaluation_cell}")
        lateral_edges = (-1.78, -0.61, 0.61, 1.78)
        speed_edges = (3.0, 5.0, 7.0, 9.0)
        height_edges = (BALL_RADIUS_M, 0.35, 0.75, 1.12)
        target_y = _uniform(
            count,
            lateral_edges[lateral_cell],
            lateral_edges[lateral_cell + 1],
            device=env.device,
        )
        forward_speed = _uniform(
            count,
            speed_edges[speed_cell],
            speed_edges[speed_cell + 1],
            device=env.device,
        )
        target_z = _uniform(
            count,
            height_edges[height_cell],
            height_edges[height_cell + 1],
            device=env.device,
        )
    if stage == 0:
        spawn_y = target_y.clone()
    else:
        spawn_y = _uniform(count, *spawn_y_ranges[stage], device=env.device)
    flight_time = spawn_x / forward_speed
    lateral_velocity = (target_y - spawn_y) / flight_time
    vertical_velocity = (
        target_z - BALL_RADIUS_M + 0.5 * 9.81 * flight_time.square()
    ) / flight_time
    if stage <= 1:
        vertical_velocity.zero_()

    ball_root = ball.data.default_root_state[ids].clone()
    ball_root[:, 0] = origins[:, 0] + spawn_x
    ball_root[:, 1] = origins[:, 1] + spawn_y
    ball_root[:, 2] = origins[:, 2] + BALL_RADIUS_M
    ball_root[:, 3:7] = 0.0
    ball_root[:, 3] = 1.0
    ball_root[:, 7] = -forward_speed
    ball_root[:, 8] = lateral_velocity
    ball_root[:, 9] = vertical_velocity
    spin = (0.0, 0.0, 3.0, 8.0)[stage]
    ball_root[:, 10:13] = _uniform(
        count * 3, -spin, spin, device=env.device
    ).view(count, 3)
    ball.write_root_state_to_sim(ball_root, env_ids=ids)

    state.initial_speed[ids] = torch.linalg.vector_norm(ball_root[:, 7:10], dim=-1)
    state.shot_forward_speed[ids] = forward_speed
    state.shot_target_y[ids] = target_y
    state.shot_target_z[ids] = target_z
    state.shot_origin_y[ids] = spawn_y
    state.disturbance_level[ids] = float(stage) / 3.0


def goalkeeper_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    stage_thresholds: tuple[int, int, int],
) -> torch.Tensor:
    del env_ids
    return torch.tensor(
        _stage_from_counter(env.common_step_counter, stage_thresholds),
        device=env.device,
        dtype=torch.float32,
    )


def ball_state_body(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    relative_position_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    relative_velocity_w = (
        ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
    )
    relative_position_b = quat_apply_inverse(
        robot.data.root_link_quat_w, relative_position_w
    )
    relative_velocity_b = quat_apply_inverse(
        robot.data.root_link_quat_w, relative_velocity_w
    )
    return torch.cat(
        (
            relative_position_b / torch.tensor(
                (6.5, 2.2, 1.4), device=env.device
            ),
            relative_velocity_b / torch.tensor(
                (9.0, 5.0, 8.0), device=env.device
            ),
        ),
        dim=-1,
    )


def _predicted_intercept(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, ...]:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    ball_pos = ball.data.root_link_pos_w
    ball_vel = ball.data.root_link_lin_vel_w
    vx = ball_vel[:, 0]
    approaching = vx < -0.05
    time_to_plane = torch.where(
        approaching,
        (robot.data.root_link_pos_w[:, 0] - ball_pos[:, 0]) / vx,
        torch.full_like(vx, 5.0),
    ).clamp(0.0, 5.0)
    predicted_y = ball_pos[:, 1] + ball_vel[:, 1] * time_to_plane
    predicted_z = (
        ball_pos[:, 2]
        + ball_vel[:, 2] * time_to_plane
        - 0.5 * 9.81 * time_to_plane.square()
    )
    return predicted_y, predicted_z, time_to_plane, approaching


def intercept_features(env: ManagerBasedRlEnv) -> torch.Tensor:
    predicted_y, predicted_z, time_to_plane, approaching = _predicted_intercept(env)
    origin_y = env.scene.env_origins[:, 1]
    return torch.stack(
        (
            (predicted_y - origin_y) / GOAL_HALF_WIDTH_M,
            predicted_z.clamp(-0.25, 1.6) / GOAL_HEIGHT_M,
            time_to_plane / 5.0,
            approaching.float(),
        ),
        dim=-1,
    )


def goalkeeper_context(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    relative = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    # Both coordinates are obtainable from deployment odometry.  Supplying the
    # longitudinal coordinate avoids asking a memoryless actor to infer whether
    # it has drifted in front of or behind its goal line from reward alone.
    return torch.stack(
        (relative[:, 0], relative[:, 1] / GOAL_HALF_WIDTH_M), dim=-1
    )


def privileged_scenario(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = goalkeeper_state(env)
    return torch.stack(
        (
            state.curriculum_stage.float() / 3.0,
            state.shot_target_y / GOAL_HALF_WIDTH_M,
            state.shot_target_z / GOAL_HEIGHT_M,
            state.shot_forward_speed / 10.0,
            state.disturbance_level,
        ),
        dim=-1,
    )


def privileged_safety_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = goalkeeper_state(env)
    return torch.stack(
        (
            state.current_legal.float(),
            state.current_head.float(),
            state.current_hand.float(),
            state.blocked.float(),
            state.fell.float(),
            state.joint_violation.float(),
            (state.torque_saturation_steps.float() / max(1, env.max_episode_length)).clamp(0.0, 1.0),
            (state.post_block_stand_steps.float() / (3.0 * CONTROL_HZ)).clamp(0.0, 1.5),
        ),
        dim=-1,
    )


def privileged_contact_forces(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = goalkeeper_state(env)
    return torch.stack(
        (
            state.peak_foot_force / 1000.0,
            state.peak_body_force / 1000.0,
            state.peak_head_force / 1000.0,
            state.peak_hand_force / 1000.0,
        ),
        dim=-1,
    )


def _sensor_peak_force(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    sensor = env.scene[name]
    history = sensor.data.force_history
    if history is not None:
        return torch.linalg.vector_norm(history, dim=-1).amax(dim=(1, 2))
    force = sensor.data.force
    if force is None:
        return _zeros(env)
    return torch.linalg.vector_norm(force, dim=-1).amax(dim=1)


def _finite_worlds(env: ManagerBasedRlEnv) -> torch.Tensor:
    data = env.sim.data
    tensors = (data.qpos, data.qvel, data.ctrl)
    result = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    for tensor in tensors:
        result &= torch.isfinite(tensor).reshape(env.num_envs, -1).all(dim=1)
    return result


class GoalkeeperTransition:
    """Update strict BallEvent-style evidence once per policy step."""

    def __init__(self, cfg: Any, env: ManagerBasedRlEnv):
        del cfg
        self.__env = env
        self._state = goalkeeper_state(env)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        # The reset event has already initialized the new scenario.  This reset is
        # intentionally idempotent and only clears the transition stamp so the
        # first physics step is always processed.
        ids = _ids(self._env, env_ids)
        self._state.last_transition_step[ids] = -1

    @property
    def _env(self) -> ManagerBasedRlEnv:
        # The state carries no environment reference, so this property is filled
        # by __call__ before reset is ever used by the manager.
        return self.__env

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        auto_terminate: bool,
        safe_after_contact_seconds: float,
        post_block_stand_seconds: float,
        head_impact_limit_n: float,
        hand_impact_limit_n: float,
        joint_limit_tolerance_rad: float,
        torque_saturation_fraction: float,
    ) -> torch.Tensor:
        self.__env = env
        state = self._state
        stamp = int(env.common_step_counter)
        if bool(torch.all(state.last_transition_step == stamp)):
            done = state.success | state.conceded | state.unsafe | state.nonfinite
            return done if auto_terminate else torch.zeros_like(done)

        foot_force = _sensor_peak_force(env, "ball_foot_calf_contact")
        body_force = _sensor_peak_force(env, "ball_legal_body_contact")
        head_force = torch.maximum(
            _sensor_peak_force(env, "ball_head_contact"),
            _sensor_peak_force(env, "head_ground_contact"),
        )
        hand_force = torch.maximum(
            _sensor_peak_force(env, "ball_hand_forearm_contact"),
            _sensor_peak_force(env, "hand_ground_contact"),
        )
        foot_now = foot_force > 1.0e-4
        body_now = body_force > 1.0e-4
        head_now = head_force > 1.0e-4
        hand_now = hand_force > 1.0e-4
        legal_now = foot_now | body_now

        previous_blocked = state.blocked.clone()
        previous_fell = state.fell.clone()
        previous_success = state.success.clone()
        state.current_legal[:] = legal_now
        state.current_head[:] = head_now
        state.current_hand[:] = hand_now
        state.ever_foot_or_calf |= foot_now
        state.ever_body |= body_now
        state.ever_head |= head_now
        state.ever_hand |= hand_now
        state.ever_legal |= legal_now
        state.peak_foot_force[:] = torch.maximum(state.peak_foot_force, foot_force)
        state.peak_body_force[:] = torch.maximum(state.peak_body_force, body_force)
        state.peak_head_force[:] = torch.maximum(state.peak_head_force, head_force)
        state.peak_hand_force[:] = torch.maximum(state.peak_hand_force, hand_force)

        robot: Entity = env.scene["robot"]
        ball: Entity = env.scene["ball"]
        origin = env.scene.env_origins
        ball_pos = ball.data.root_link_pos_w - origin
        ball_vel = ball.data.root_link_lin_vel_w
        speed = torch.linalg.vector_norm(ball_vel, dim=-1)
        predicted_y, predicted_z, _, approaching = _predicted_intercept(env)
        predicted_y -= origin[:, 1]
        predicted_outside = (
            predicted_y.abs() > GOAL_HALF_WIDTH_M + BALL_RADIUS_M
        ) | (predicted_z > GOAL_HEIGHT_M + BALL_RADIUS_M)
        ball_safe = (~approaching) | (speed < 0.40) | predicted_outside
        safe_now = state.ever_legal & ball_safe & (~state.conceded)
        state.safe_steps[:] = torch.where(
            safe_now, state.safe_steps + 1, torch.zeros_like(state.safe_steps)
        )
        safe_steps_required = max(1, round(safe_after_contact_seconds / env.step_dt))
        state.blocked[:] = state.ever_legal & (state.safe_steps >= safe_steps_required)
        state.new_block[:] = state.blocked & (~previous_blocked)

        conceded_now = (
            (ball_pos[:, 0] <= GOAL_LINE_X_M)
            & (ball_pos[:, 1].abs() <= GOAL_HALF_WIDTH_M + BALL_RADIUS_M)
            & (ball_pos[:, 2] <= GOAL_HEIGHT_M + BALL_RADIUS_M)
        )
        state.conceded |= conceded_now

        upright_cos = -robot.data.projected_gravity_b[:, 2]
        standing = (robot.data.root_link_pos_w[:, 2] - origin[:, 2] > 0.53) & (
            upright_cos > 0.75
        )
        fell_now = (robot.data.root_link_pos_w[:, 2] - origin[:, 2] < 0.34) | (
            upright_cos < 0.35
        )
        state.fell |= fell_now
        state.new_fall[:] = state.fell & (~previous_fell)
        state.post_block_stand_steps[:] = torch.where(
            state.blocked & standing,
            state.post_block_stand_steps + 1,
            torch.zeros_like(state.post_block_stand_steps),
        )

        joint_pos = robot.data.joint_pos
        limits = robot.data.joint_pos_limits
        state.joint_violation |= (
            (joint_pos < limits[..., 0] - joint_limit_tolerance_rad)
            | (joint_pos > limits[..., 1] + joint_limit_tolerance_rad)
        ).any(dim=1)
        caps = torch.tensor(T1_EFFECTIVE_TORQUE_CAPS, device=env.device).view(1, -1)
        saturation_now = (
            robot.data.qfrc_actuator.abs() >= caps * torque_saturation_fraction
        ).any(dim=1)
        state.torque_saturation_steps += saturation_now.long()
        state.nonfinite |= ~_finite_worlds(env)
        state.unsafe |= (state.peak_head_force > head_impact_limit_n) | (
            state.peak_hand_force > hand_impact_limit_n
        )

        stand_steps_required = max(1, round(post_block_stand_seconds / env.step_dt))
        state.success[:] = (
            state.blocked
            & (state.post_block_stand_steps >= stand_steps_required)
            & (~state.conceded)
            & (~state.fell)
            & (~state.joint_violation)
            & (~state.unsafe)
            & (~state.nonfinite)
        )
        state.new_success[:] = state.success & (~previous_success)
        state.last_transition_step[:] = stamp
        done = state.success | state.conceded | state.unsafe | state.nonfinite
        return done if auto_terminate else torch.zeros_like(done)


def intercept_alignment_reward(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    predicted_y, _, _, approaching = _predicted_intercept(env)
    # A small foot offset discourages the ball from passing between the feet.
    relative_prediction = predicted_y - env.scene.env_origins[:, 1]
    foot_offset = torch.where(
        relative_prediction >= 0.0,
        torch.full_like(relative_prediction, -0.16),
        torch.full_like(relative_prediction, 0.16),
    )
    desired_y = predicted_y + foot_offset
    error = robot.data.root_link_pos_w[:, 1] - desired_y
    return torch.exp(-0.5 * (error / std).square()) * approaching.float()


def goal_line_position_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    relative = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.exp(-0.5 * (relative[:, 0] / 0.35).square()) * torch.exp(
        -0.5 * (relative[:, 1] / 2.0).square()
    )


def upright_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    return ((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0) + 1.0) * 0.5


def legal_contact_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).current_legal.float()


def new_block_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).new_block.float()


def post_block_standing_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = goalkeeper_state(env)
    return (state.blocked & (~state.fell)).float()


def competitive_success_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).new_success.float()


def concede_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).conceded.float()


def unsafe_impact_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).unsafe.float()


def new_fall_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).new_fall.float()


def dangerous_force_cost(
    env: ManagerBasedRlEnv, head_limit_n: float, hand_limit_n: float
) -> torch.Tensor:
    state = goalkeeper_state(env)
    head = torch.relu(state.peak_head_force - head_limit_n) / head_limit_n
    hand = torch.relu(state.peak_hand_force - hand_limit_n) / hand_limit_n
    return head + hand


def torque_saturation_cost(env: ManagerBasedRlEnv, fraction: float) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    caps = torch.tensor(T1_EFFECTIVE_TORQUE_CAPS, device=env.device).view(1, -1)
    utilization = robot.data.qfrc_actuator.abs() / caps
    return torch.relu(utilization - fraction).mean(dim=1)


# Metric accessors intentionally expose distinct facts.  The evaluation script
# combines them under a fail-closed fixed/held-out promotion contract.
def metric_success(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).success.float()


def metric_blocked(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).blocked.float()


def metric_conceded(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).conceded.float()


def metric_legal_touch(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).ever_legal.float()


def metric_foot_touch(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).ever_foot_or_calf.float()


def metric_body_touch(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).ever_body.float()


def metric_head_touch(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).ever_head.float()


def metric_hand_touch(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).ever_hand.float()


def metric_fell(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).fell.float()


def metric_joint_violation(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).joint_violation.float()


def metric_unsafe(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).unsafe.float()


def metric_nonfinite(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).nonfinite.float()


def metric_torque_saturation_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = goalkeeper_state(env)
    return state.torque_saturation_steps.float() / torch.clamp(
        env.episode_length_buf.float(), min=1.0
    )


def metric_post_block_stand_seconds(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).post_block_stand_steps.float() * env.step_dt


def metric_peak_head_force(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).peak_head_force


def metric_peak_hand_force(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).peak_hand_force


def metric_curriculum_stage(env: ManagerBasedRlEnv) -> torch.Tensor:
    return goalkeeper_state(env).curriculum_stage.float()
