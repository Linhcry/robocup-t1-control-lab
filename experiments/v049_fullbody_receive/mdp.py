"""GPU-vectorized MDP terms for v049 physical receiving and ball trapping.

The policy actor sees deployable proprioception and a delayed/noisy relative-ball
estimate.  Exact simulator state, contact latches, curriculum labels and safety
bookkeeping are critic/evaluation-only.  A successful receive requires a real
foot-or-shank collision followed by continuous physical control of the ball.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from experiments.v044_mjlab_t1_getup.rcss_getup_controller import (
    RCSS_EFFECTIVE_EFFORT_LIMIT,
)
from locomotion.t1_spec import T1_JOINT_NAMES, T1_LOWER_LIMITS, T1_UPPER_LIMITS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


ACTION_DIM = 23
ACTOR_OBS_DIM = 81
CRITIC_OBS_DIM = 109
CONTROL_HZ = 50.0
BALL_RADIUS_M = 0.11
BALL_MASS_KG = 0.41
HOLD_SECONDS = 1.0
CONTROL_RADIUS_M = 0.55
MAX_CONTROLLED_BALL_SPEED_MPS = 0.65
MAX_CONTROLLED_RELATIVE_SPEED_MPS = 0.45
MINIMUM_SPEED_REDUCTION_FRACTION = 0.50
DANGEROUS_GROUND_FORCE_N = 350.0
STAGE_THRESHOLDS = (600 * 32, 1500 * 32, 2600 * 32)


def _canonical_name(name: str) -> str:
    return "Head_yaw" if name == "AAHead_yaw" else name


class SafeT1JointPositionAction(BaseAction):
    """Fail closed on shape/non-finite actions and clamp all 23 joint targets."""

    cfg: "SafeT1JointPositionActionCfg"

    def __init__(self, cfg: "SafeT1JointPositionActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)
        names = tuple(_canonical_name(name) for name in self._target_names)
        if self.action_dim != ACTION_DIM or names != tuple(T1_JOINT_NAMES):
            raise ValueError(
                "v049 action order must be canonical RCSS T1 23-DoF; "
                f"received {self.action_dim} targets: {names}"
            )
        xml_limits = self._entity.data.joint_pos_limits[:, self._target_ids]
        canonical_lower = torch.as_tensor(
            T1_LOWER_LIMITS, device=self.device, dtype=torch.float32
        ).view(1, ACTION_DIM)
        canonical_upper = torch.as_tensor(
            T1_UPPER_LIMITS, device=self.device, dtype=torch.float32
        ).view(1, ACTION_DIM)
        self._lower = torch.maximum(canonical_lower, xml_limits[..., 0])
        self._upper = torch.minimum(canonical_upper, xml_limits[..., 1])
        if not bool(torch.all(self._lower < self._upper)):
            raise ValueError("v049 canonical and RCSS XML joint limits do not overlap")
        self._offset = self._entity.data.default_joint_pos[:, self._target_ids].clone()

    @property
    def clamped_target(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(
                f"v049 action must have shape ({self.num_envs}, {ACTION_DIM}), "
                f"received {tuple(actions.shape)}"
            )
        if not bool(torch.isfinite(actions).all()):
            raise FloatingPointError("v049 rejected a NaN/Inf policy action")
        self._raw_actions.copy_(actions)
        target = torch.clamp(actions, -1.0, 1.0) * self._scale + self._offset
        self._processed_actions = torch.maximum(
            self._lower, torch.minimum(self._upper, target)
        )

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = self._processed_actions - encoder_bias
        target = torch.maximum(self._lower, torch.minimum(self._upper, target))
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class SafeT1JointPositionActionCfg(BaseActionCfg):
    scale: float = 0.55
    use_default_offset: bool = True
    preserve_order: bool = True

    def build(self, env: "ManagerBasedRlEnv") -> SafeT1JointPositionAction:
        if not self.use_default_offset:
            raise ValueError("v049 requires the audited RCSS default-pose offset")
        return SafeT1JointPositionAction(self, env)


def _zeros(env: "ManagerBasedRlEnv", *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=dtype)


def _ids(
    env: "ManagerBasedRlEnv", env_ids: torch.Tensor | slice | None
) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def receive_state(env: "ManagerBasedRlEnv") -> SimpleNamespace:
    existing = getattr(env, "_v049_receive_state", None)
    if existing is not None:
        return existing
    state = SimpleNamespace(
        curriculum_stage=_zeros(env, dtype=torch.long),
        launch_speed=_zeros(env),
        launch_angle_rad=_zeros(env),
        target_y=_zeros(env),
        target_z=_zeros(env),
        initial_ball_speed=_zeros(env),
        previous_ball_speed=_zeros(env),
        first_legal_contact_speed=_zeros(env),
        ball_speed=_zeros(env),
        relative_speed=_zeros(env),
        ball_distance=_zeros(env),
        speed_reduction=_zeros(env),
        hold_steps=_zeros(env, dtype=torch.long),
        saturation_steps=_zeros(env, dtype=torch.long),
        legal_touched=_zeros(env, dtype=torch.bool),
        illegal_touched=_zeros(env, dtype=torch.bool),
        dangerous_contact=_zeros(env, dtype=torch.bool),
        fell=_zeros(env, dtype=torch.bool),
        joint_violation=_zeros(env, dtype=torch.bool),
        torque_violation=_zeros(env, dtype=torch.bool),
        nonfinite=_zeros(env, dtype=torch.bool),
        missed=_zeros(env, dtype=torch.bool),
        stable_now=_zeros(env, dtype=torch.bool),
        success=_zeros(env, dtype=torch.bool),
        new_success=_zeros(env, dtype=torch.bool),
        peak_legal_force=_zeros(env),
        peak_illegal_force=_zeros(env),
        peak_dangerous_ground_force=_zeros(env),
        last_update_step=torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        ),
    )
    env._v049_receive_state = state
    return state


_BOOL_FIELDS = (
    "legal_touched",
    "illegal_touched",
    "dangerous_contact",
    "fell",
    "joint_violation",
    "torque_violation",
    "nonfinite",
    "missed",
    "stable_now",
    "success",
    "new_success",
)
_FLOAT_FIELDS = (
    "launch_speed",
    "launch_angle_rad",
    "target_y",
    "target_z",
    "initial_ball_speed",
    "previous_ball_speed",
    "first_legal_contact_speed",
    "ball_speed",
    "relative_speed",
    "ball_distance",
    "speed_reduction",
    "peak_legal_force",
    "peak_illegal_force",
    "peak_dangerous_ground_force",
)


def _reset_state(env: "ManagerBasedRlEnv", ids: torch.Tensor) -> SimpleNamespace:
    state = receive_state(env)
    for name in _BOOL_FIELDS:
        getattr(state, name)[ids] = False
    for name in _FLOAT_FIELDS:
        getattr(state, name)[ids] = 0.0
    state.curriculum_stage[ids] = 0
    state.hold_steps[ids] = 0
    state.saturation_steps[ids] = 0
    state.last_update_step[ids] = -1
    return state


def _uniform(
    count: int, low: float, high: float, *, device: str
) -> torch.Tensor:
    if low == high:
        return torch.full((count,), low, device=device)
    return low + (high - low) * torch.rand(count, device=device)


def stage_from_counter(counter: int, thresholds: tuple[int, int, int]) -> int:
    return int(counter >= thresholds[0]) + int(counter >= thresholds[1]) + int(
        counter >= thresholds[2]
    )


def reset_receive_scenario(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | slice,
    stage_thresholds: tuple[int, int, int],
    stage_override: int | None = None,
    evaluation_cell: tuple[int, int, int] | None = None,
) -> None:
    """Reset robot and launch a real ball toward a sampled receiving point.

    The three curriculum axes are incoming speed, horizontal arrival angle and
    physical intercept/landing point.  ``evaluation_cell`` fixes a 3x3x3 cell
    without changing the success contract and is used by strict evaluation.
    """

    ids = _ids(env, env_ids)
    count = len(ids)
    state = _reset_state(env, ids)
    stage = (
        stage_from_counter(env.common_step_counter, stage_thresholds)
        if stage_override is None
        else int(stage_override)
    )
    if stage not in (0, 1, 2, 3):
        raise ValueError(f"v049 curriculum stage must be 0..3, received {stage}")
    state.curriculum_stage[ids] = stage

    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    origins = env.scene.env_origins[ids]
    robot_root = robot.data.default_root_state[ids].clone()
    robot_root[:, :3] += origins
    yaw_jitter = (0.0, 0.02, 0.05, 0.10)[stage]
    yaw = _uniform(count, -yaw_jitter, yaw_jitter, device=env.device)
    robot_root[:, 3:7] = 0.0
    robot_root[:, 3] = torch.cos(0.5 * yaw)
    robot_root[:, 6] = torch.sin(0.5 * yaw)
    robot_root[:, 7:13] = 0.0
    robot.write_root_state_to_sim(robot_root, env_ids=ids)

    joint_pos = robot.data.default_joint_pos[ids].clone()
    if stage >= 2:
        jitter = (0.0, 0.0, 0.008, 0.015)[stage]
        joint_pos += _uniform(
            count * ACTION_DIM, -jitter, jitter, device=env.device
        ).view(count, ACTION_DIM)
    limits = robot.data.joint_pos_limits[ids]
    joint_pos = torch.maximum(limits[..., 0], torch.minimum(limits[..., 1], joint_pos))
    robot.write_joint_state_to_sim(
        joint_pos, torch.zeros_like(joint_pos), env_ids=ids
    )

    speed_ranges = ((0.9, 1.4), (1.0, 2.2), (1.2, 3.3), (1.0, 4.5))
    angle_ranges_deg = ((-4.0, 4.0), (-14.0, 14.0), (-28.0, 28.0), (-45.0, 45.0))
    target_y_ranges = ((-0.06, 0.06), (-0.18, 0.18), (-0.34, 0.34), (-0.52, 0.52))
    target_z_ranges = (
        (BALL_RADIUS_M, BALL_RADIUS_M),
        (BALL_RADIUS_M, BALL_RADIUS_M),
        (BALL_RADIUS_M, 0.20),
        (BALL_RADIUS_M, 0.30),
    )
    spawn_x_ranges = ((1.35, 1.75), (1.55, 2.15), (1.8, 2.7), (2.0, 3.2))

    if evaluation_cell is None:
        speed = _uniform(count, *speed_ranges[stage], device=env.device)
        angle_deg = _uniform(count, *angle_ranges_deg[stage], device=env.device)
        target_y = _uniform(count, *target_y_ranges[stage], device=env.device)
        target_z = _uniform(count, *target_z_ranges[stage], device=env.device)
    else:
        speed_cell, angle_cell, landing_cell = (int(v) for v in evaluation_cell)
        if not all(0 <= value < 3 for value in evaluation_cell):
            raise ValueError(f"invalid v049 evaluation cell: {evaluation_cell}")
        speed_edges = (1.0, 2.0, 3.0, 4.5)
        angle_edges = (-45.0, -15.0, 15.0, 45.0)
        landing_edges = (-0.52, -0.17, 0.17, 0.52)
        speed = _uniform(
            count, speed_edges[speed_cell], speed_edges[speed_cell + 1], device=env.device
        )
        angle_deg = _uniform(
            count, angle_edges[angle_cell], angle_edges[angle_cell + 1], device=env.device
        )
        target_y = _uniform(
            count,
            landing_edges[landing_cell],
            landing_edges[landing_cell + 1],
            device=env.device,
        )
        target_z = _uniform(count, BALL_RADIUS_M, 0.30, device=env.device)

    angle = torch.deg2rad(angle_deg)
    spawn_x = _uniform(count, *spawn_x_ranges[stage], device=env.device)
    target_x = torch.full_like(spawn_x, 0.24)
    vx = -speed * torch.cos(angle).clamp_min(0.70)
    vy = speed * torch.sin(angle)
    flight_time = (spawn_x - target_x) / (-vx).clamp_min(0.25)
    spawn_y = target_y - vy * flight_time
    spawn_z = torch.full_like(spawn_x, BALL_RADIUS_M + 0.003)
    vz = torch.zeros_like(speed)
    airborne = target_z > BALL_RADIUS_M + 0.005
    solved_vz = (
        target_z - spawn_z + 0.5 * 9.81 * flight_time.square()
    ) / flight_time.clamp_min(0.1)
    vz = torch.where(airborne, solved_vz, vz)

    root = ball.data.default_root_state[ids].clone()
    root[:, 0] = origins[:, 0] + spawn_x
    root[:, 1] = origins[:, 1] + spawn_y
    root[:, 2] = origins[:, 2] + spawn_z
    root[:, 3:7] = 0.0
    root[:, 3] = 1.0
    root[:, 7] = vx
    root[:, 8] = vy
    root[:, 9] = vz
    spin_limit = (0.0, 1.0, 4.0, 8.0)[stage]
    root[:, 10:13] = _uniform(
        count * 3, -spin_limit, spin_limit, device=env.device
    ).view(count, 3)
    ball.write_root_state_to_sim(root, env_ids=ids)

    initial_speed = torch.linalg.vector_norm(root[:, 7:10], dim=-1)
    state.launch_speed[ids] = speed
    state.launch_angle_rad[ids] = angle
    state.target_y[ids] = target_y
    state.target_z[ids] = target_z
    state.initial_ball_speed[ids] = initial_speed
    state.previous_ball_speed[ids] = initial_speed


def receive_curriculum(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | slice,
    stage_thresholds: tuple[int, int, int],
) -> torch.Tensor:
    del env_ids
    return torch.tensor(
        stage_from_counter(env.common_step_counter, stage_thresholds),
        device=env.device,
        dtype=torch.float32,
    )


def _sensor_peak_force(env: "ManagerBasedRlEnv", name: str) -> torch.Tensor:
    sensor = env.scene[name]
    history = sensor.data.force_history
    if history is not None:
        return torch.linalg.vector_norm(history, dim=-1).amax(dim=(1, 2))
    force = sensor.data.force
    if force is None:
        return _zeros(env)
    return torch.linalg.vector_norm(force, dim=-1).amax(dim=1)


def _finite_worlds(env: "ManagerBasedRlEnv") -> torch.Tensor:
    result = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    for tensor in (env.sim.data.qpos, env.sim.data.qvel, env.sim.data.ctrl):
        result &= torch.isfinite(tensor).reshape(env.num_envs, -1).all(dim=1)
    return result


def update_receive_state(env: "ManagerBasedRlEnv") -> SimpleNamespace:
    """Update real-contact, control-duration and safety evidence once per step."""

    state = receive_state(env)
    stamp = int(env.common_step_counter)
    needs = state.last_update_step != stamp
    if not bool(needs.any()):
        return state

    legal_force = _sensor_peak_force(env, "legal_lower_leg_ball_contact")
    illegal_force = _sensor_peak_force(env, "illegal_robot_ball_contact")
    danger_force = _sensor_peak_force(env, "dangerous_ground_contact")
    legal_now = legal_force > 1.0e-4
    illegal_now = illegal_force > 1.0e-4
    danger_now = danger_force > DANGEROUS_GROUND_FORCE_N

    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    origin = env.scene.env_origins
    ball_speed = torch.linalg.vector_norm(ball.data.root_link_lin_vel_w, dim=-1)
    relative_velocity = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
    relative_speed = torch.linalg.vector_norm(relative_velocity[:, :2], dim=-1)
    distance = torch.linalg.vector_norm(
        ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=-1
    )

    first_legal = needs & legal_now & (~state.legal_touched)
    contact_reference = torch.maximum(state.previous_ball_speed, ball_speed)
    state.first_legal_contact_speed[first_legal] = contact_reference[first_legal]
    state.legal_touched[needs] |= legal_now[needs]
    state.illegal_touched[needs] |= illegal_now[needs]
    state.dangerous_contact[needs] |= danger_now[needs]
    state.peak_legal_force[needs] = torch.maximum(
        state.peak_legal_force[needs], legal_force[needs]
    )
    state.peak_illegal_force[needs] = torch.maximum(
        state.peak_illegal_force[needs], illegal_force[needs]
    )
    state.peak_dangerous_ground_force[needs] = torch.maximum(
        state.peak_dangerous_ground_force[needs], danger_force[needs]
    )
    state.ball_speed[needs] = ball_speed[needs]
    state.relative_speed[needs] = relative_speed[needs]
    state.ball_distance[needs] = distance[needs]

    reference = state.first_legal_contact_speed.clamp_min(1.0e-4)
    reduction = torch.where(
        state.legal_touched,
        (reference - ball_speed) / reference,
        torch.zeros_like(ball_speed),
    )
    state.speed_reduction[needs] = reduction[needs]

    root_height = robot.data.root_link_pos_w[:, 2] - origin[:, 2]
    upright_cos = -robot.data.projected_gravity_b[:, 2]
    fallen_now = (root_height < 0.48) | (upright_cos < 0.68)
    state.fell[needs] |= fallen_now[needs]
    limits = robot.data.joint_pos_limits
    joint_now = (
        (robot.data.joint_pos < limits[..., 0] - 1.0e-4)
        | (robot.data.joint_pos > limits[..., 1] + 1.0e-4)
    ).any(dim=1)
    state.joint_violation[needs] |= joint_now[needs]
    caps = torch.as_tensor(
        RCSS_EFFECTIVE_EFFORT_LIMIT,
        device=env.device,
        dtype=robot.data.qfrc_actuator.dtype,
    ).view(1, ACTION_DIM)
    torque_now = (
        robot.data.qfrc_actuator.abs() > caps * (1.0 + 1.0e-3)
    ).any(dim=1)
    state.torque_violation[needs] |= torque_now[needs]
    saturated = (robot.data.qfrc_actuator.abs() >= caps * 0.98).any(dim=1)
    state.saturation_steps[needs] += saturated[needs].long()
    state.nonfinite[needs] |= (~_finite_worlds(env))[needs]

    local_ball = ball.data.root_link_pos_w - origin
    missed_now = (
        (local_ball[:, 0] < -1.20)
        | (local_ball[:, :2].abs() > 5.0).any(dim=1)
        | (local_ball[:, 2] < -0.05)
        | (local_ball[:, 2] > 2.0)
    )
    state.missed[needs] |= missed_now[needs]

    contact_was_meaningful = state.first_legal_contact_speed >= torch.maximum(
        torch.full_like(state.initial_ball_speed, 0.55),
        state.initial_ball_speed * 0.45,
    )
    controlled_height = (local_ball[:, 2] >= BALL_RADIUS_M * 0.65) & (
        local_ball[:, 2] <= 0.35
    )
    safe = ~(
        state.illegal_touched
        | state.dangerous_contact
        | state.fell
        | state.joint_violation
        | state.torque_violation
        | state.nonfinite
        | state.missed
    )
    stable = (
        state.legal_touched
        & contact_was_meaningful
        & (reduction >= MINIMUM_SPEED_REDUCTION_FRACTION)
        & (ball_speed <= MAX_CONTROLLED_BALL_SPEED_MPS)
        & (relative_speed <= MAX_CONTROLLED_RELATIVE_SPEED_MPS)
        & (distance <= CONTROL_RADIUS_M)
        & controlled_height
        & safe
    )
    state.stable_now[needs] = stable[needs]
    reset_hold = needs & (~stable)
    add_hold = needs & stable
    state.hold_steps[reset_hold] = 0
    state.hold_steps[add_hold] += 1
    previous_success = state.success.clone()
    required_steps = max(1, round(HOLD_SECONDS / env.step_dt))
    achieved = (state.hold_steps >= required_steps) & safe
    state.success[needs] |= achieved[needs]
    state.new_success[needs] = state.success[needs] & (~previous_success[needs])
    state.previous_ball_speed[needs] = ball_speed[needs]
    state.last_update_step[needs] = stamp
    return state


def receive_transition(env: "ManagerBasedRlEnv", auto_terminate: bool = True) -> torch.Tensor:
    state = update_receive_state(env)
    failure = (
        state.illegal_touched
        | state.dangerous_contact
        | state.fell
        | state.joint_violation
        | state.torque_violation
        | state.nonfinite
        | state.missed
    )
    done = state.success | failure
    return done if auto_terminate else torch.zeros_like(done)


# Actor terms: relative state can be provided by cameras/state estimation on the
# robot. Observation configuration adds noise and delay. No contact latch,
# curriculum stage, launch truth, world pose or safety flag is exposed here.
def ball_relative_position_b(env: "ManagerBasedRlEnv") -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    delta = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    return quat_apply(quat_inv(robot.data.root_link_quat_w), delta)


def ball_relative_velocity_b(env: "ManagerBasedRlEnv") -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    delta = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
    return quat_apply(quat_inv(robot.data.root_link_quat_w), delta)


# Critic-only truth.
def privileged_ball_world_state(env: "ManagerBasedRlEnv") -> torch.Tensor:
    ball: Entity = env.scene["ball"]
    local = ball.data.root_link_pos_w - env.scene.env_origins
    return torch.cat((local, ball.data.root_link_vel_w), dim=-1)


def privileged_scenario(env: "ManagerBasedRlEnv") -> torch.Tensor:
    state = receive_state(env)
    return torch.stack(
        (
            state.launch_speed / 4.5,
            state.launch_angle_rad / (math.pi / 4.0),
            state.target_y / 0.52,
            state.target_z / 0.30,
            state.initial_ball_speed / 6.0,
            state.curriculum_stage.float() / 3.0,
        ),
        dim=-1,
    )


def privileged_receive_state(env: "ManagerBasedRlEnv") -> torch.Tensor:
    state = receive_state(env)
    return torch.stack(
        (
            state.legal_touched.float(),
            state.illegal_touched.float(),
            state.dangerous_contact.float(),
            state.fell.float(),
            state.joint_violation.float(),
            state.torque_violation.float(),
            state.ball_speed / 6.0,
            state.ball_distance / CONTROL_RADIUS_M,
            state.speed_reduction,
            state.hold_steps.float() / CONTROL_HZ,
        ),
        dim=-1,
    )


# Reward terms.
def upright_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    return ((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0) + 1.0) * 0.5


def intercept_alignment_reward(env: "ManagerBasedRlEnv", std_m: float = 0.32) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    pos = ball.data.root_link_pos_w
    vel = ball.data.root_link_lin_vel_w
    receiver_x = robot.data.root_link_pos_w[:, 0] + 0.24
    time = ((receiver_x - pos[:, 0]) / vel[:, 0]).clamp(min=0.0, max=3.0)
    predicted_y = pos[:, 1] + vel[:, 1] * time
    lateral_error = robot.data.root_link_pos_w[:, 1] - predicted_y
    approaching = vel[:, 0] < -0.05
    return torch.exp(-0.5 * torch.square(lateral_error / std_m)) * approaching.float()


def approach_control_radius_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    state = update_receive_state(env)
    return torch.exp(-torch.square(state.ball_distance / CONTROL_RADIUS_M))


def legal_contact_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return (_sensor_peak_force(env, "legal_lower_leg_ball_contact") > 1.0e-4).float()


def speed_reduction_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    state = update_receive_state(env)
    return state.legal_touched.float() * state.speed_reduction.clamp(0.0, 1.0)


def stable_control_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).stable_now.float()


def success_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).new_success.float()


def illegal_contact_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return (_sensor_peak_force(env, "illegal_robot_ball_contact") > 1.0e-4).float()


def danger_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).dangerous_contact.float()


def fall_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).fell.float()


def torque_saturation_cost(env: "ManagerBasedRlEnv") -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    caps = torch.as_tensor(
        RCSS_EFFECTIVE_EFFORT_LIMIT,
        device=env.device,
        dtype=robot.data.qfrc_actuator.dtype,
    ).view(1, ACTION_DIM)
    return torch.relu(robot.data.qfrc_actuator.abs() / caps - 0.90).mean(dim=1)


# Independent evidence terms used by the evaluator.
def metric_success(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).success.float()


def metric_legal_touch(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).legal_touched.float()


def metric_illegal_touch(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).illegal_touched.float()


def metric_dangerous_contact(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).dangerous_contact.float()


def metric_fell(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).fell.float()


def metric_joint_violation(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).joint_violation.float()


def metric_torque_violation(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).torque_violation.float()


def metric_hold_seconds(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).hold_steps.float() * env.step_dt


def metric_speed_reduction(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).speed_reduction


def metric_control_distance(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).ball_distance


def metric_peak_illegal_force(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).peak_illegal_force


def metric_peak_dangerous_force(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return update_receive_state(env).peak_dangerous_ground_force


def metric_curriculum_stage(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return receive_state(env).curriculum_stage.float()
