"""GPU-parallel two-T1 legal-tackle foundation on the exact RCSS model.

The defender is the only learned agent and outputs all 23 joint targets.  The
attacker is a bounded physics teacher: its joints are held at the RCSS home pose
and a capped horizontal wrench tracks a randomized approach velocity.  This is
deliberately a foundation curriculum, not a claim that a learned walking
opponent has already been solved.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz
from mjlab.viewer import ViewerConfig

from locomotion.t1_spec import (
    T1_JOINT_NAMES,
    T1_LOWER_LIMITS,
    T1_UPPER_LIMITS,
)
from experiments.v048_fullbody_legal_tackle.safe_actions import (
    V048SafeT1JointPositionActionCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


TASK_ID = "Mjlab-Fullbody-Legal-Tackle-RCSS-T1-v048"
ACTION_DIM = 23
ACTOR_OBSERVATION_DIM = 93
RCSS_JOINT_NAMES = ("AAHead_yaw", *T1_JOINT_NAMES[1:])
MINIMUM_CLEARANCE_M = 0.25
MINIMUM_CONTROL_SECONDS = 2.0
JOINT_LIMIT_TOLERANCE_RAD = 1.0e-4
TORQUE_RATIO_TOLERANCE = 1.001

# This is the executable observation contract used by both the policy and the
# v045 -> v048 warm-start audit.  Keeping the slices beside the task prevents a
# checkpoint transformer from silently assuming an obsolete concatenation
# order.  The first 78 dimensions are proprioception; football/opponent inputs
# start at column 78.
V048_ACTOR_OBSERVATION_LAYOUT = (
    ("projected_gravity", (0, 3)),
    ("base_ang_vel", (3, 6)),
    ("base_lin_vel", (6, 9)),
    ("joint_pos", (9, 32)),
    ("joint_vel", (32, 55)),
    ("last_action", (55, 78)),
    ("ball_relative_position", (78, 81)),
    ("ball_relative_velocity", (81, 84)),
    ("attacker_relative_position", (84, 87)),
    ("attacker_relative_velocity", (87, 90)),
    ("attack_direction", (90, 92)),
    ("episode_phase", (92, 93)),
)

HOME_JOINT_POS = {
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Yaw": 0.4,
    ".*_Hip_Pitch": -0.2,
    ".*_Knee_Pitch": 0.4,
    ".*_Ankle_Pitch": -0.2,
}

EFFORT_LIMITS = torch.tensor(
    [
        7.0,
        7.0,
        *([18.0] * 8),
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
    ],
    dtype=torch.float32,
)


def _find_rcss_t1_xml() -> Path:
    configured = os.environ.get("RCSS_T1_XML")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        Path("/home/linh/robot_soccer/envs/rl-mujoco").glob(
            "lib/python*/site-packages/rcsssmj/resources/robots/T1/robot.xml"
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("RCSS T1 robot.xml not found; set RCSS_T1_XML")


RCSS_T1_XML = _find_rcss_t1_xml()


def get_exact_rcss_t1_spec() -> mujoco.MjSpec:
    """Keep exact RCSS bodies/geoms/joints and replace only control ports."""

    spec = mujoco.MjSpec.from_file(str(RCSS_T1_XML))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def _position_actuator(
    names: tuple[str, ...], stiffness: float, damping: float, effort: float
) -> BuiltinPositionActuatorCfg:
    return BuiltinPositionActuatorCfg(
        target_names_expr=names,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort,
        armature=0.01,
        frictionloss=0.1,
    )


RCSS_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        _position_actuator(("AAHead_yaw", "Head_pitch"), 1.776528788, 0.226194671, 7.0),
        _position_actuator(
            (".*_Shoulder_Pitch", ".*_Shoulder_Roll", ".*_Elbow_Pitch", ".*_Elbow_Yaw"),
            27.884395859,
            3.550351553,
            36.0,
        ),
        _position_actuator(("Waist", ".*_Hip_Roll", ".*_Hip_Yaw"), 47.189045935, 6.008295943, 40.0),
        _position_actuator((".*_Hip_Pitch",), 51.707646907, 6.583622088, 55.0),
        _position_actuator((".*_Knee_Pitch",), 62.7718682, 7.992362498, 65.0),
        _position_actuator((".*_Ankle_Pitch", ".*_Ankle_Roll"), 33.512439059, 4.26693627, 50.0),
    ),
    soft_joint_pos_limit_factor=0.9,
)


HOME_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.665),
    joint_pos=HOME_JOINT_POS,
    joint_vel={".*": 0.0},
)


def get_exact_rcss_t1_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_STATE,
        spec_fn=get_exact_rcss_t1_spec,
        articulation=RCSS_ARTICULATION,
        collisions=(),
    )


def get_rcss_ball_spec() -> mujoco.MjSpec:
    """Reproduce the RCSS soccer ball's physical attributes exactly."""

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball", pos=(0.0, 0.0, 0.11))
    body.add_freejoint(name="ball-root")
    body.add_geom(
        name="ball",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.11, 0.0, 0.0),
        mass=0.41,
        friction=(0.4, 0.01, 0.01),
        condim=6,
        priority=1,
        solref=(-5000.0, -20.0),
        rgba=(1.0, 1.0, 1.0, 1.0),
    )
    return spec


BALL_STATE = EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.11))


def _add_exact_pitch(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(
        name="pitch",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0.0, 0.0, 0.0),
        size=(32.0, 24.0, 40.0),
    )


@dataclass(kw_only=True)
class ScriptedAttackerActionCfg(ActionTermCfg):
    velocity_gain: float = 45.0
    maximum_force_n: float = 35.0

    def build(self, env: ManagerBasedRlEnv) -> "ScriptedAttackerAction":
        return ScriptedAttackerAction(self, env)


class ScriptedAttackerAction(ActionTerm):
    """Zero-policy-dimension physics teacher for the attacking T1."""

    cfg: ScriptedAttackerActionCfg

    def __init__(self, cfg: ScriptedAttackerActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.empty((env.num_envs, 0), device=env.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"attacker teacher expected {self._raw_actions.shape}, got {actions.shape}")

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        del env_ids

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._entity.data.default_joint_pos)
        direction = getattr(
            self._env,
            "_v048_attack_direction",
            torch.tensor([[1.0, 0.0]], device=self.device).repeat(self.num_envs, 1),
        )
        speed = getattr(
            self._env,
            "_v048_attack_speed",
            torch.full((self.num_envs,), 0.3, device=self.device),
        )
        desired = direction * speed.unsqueeze(-1)
        velocity = self._entity.data.root_link_lin_vel_w[:, :2]
        force_xy = (desired - velocity) * float(self.cfg.velocity_gain)
        norm = torch.linalg.vector_norm(force_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
        force_xy = force_xy * torch.clamp(float(self.cfg.maximum_force_n) / norm, max=1.0)
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        torques = torch.zeros_like(forces)
        forces[:, 0, :2] = force_xy
        self._entity.write_external_wrench_to_sim(forces, torques, body_ids=[0])


def _ensure_state(env: ManagerBasedRlEnv) -> None:
    n = env.num_envs
    device = env.device
    if not hasattr(env, "_v048_attack_direction"):
        env._v048_attack_direction = torch.tensor([[1.0, 0.0]], device=device).repeat(n, 1)
        env._v048_attack_speed = torch.full((n,), 0.3, device=device)
        # 0=no legal ball contact yet, 1=attacker first, 2=defender first,
        # 3=both first appeared in the same policy interval. A transfer is
        # eligible only when this immutable episode-level owner is 1.
        env._v048_first_touch_owner = torch.zeros(
            n, dtype=torch.int8, device=device
        )
        env._v048_attacker_first_touch = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        env._v048_last_toucher = torch.zeros(n, dtype=torch.int8, device=device)
        env._v048_attacker_touched = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_defender_touched = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_turnover = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_turnover_projection = torch.zeros(n, device=device)
        env._v048_max_clearance = torch.zeros(n, device=device)
        env._v048_clearance_delta = torch.zeros(n, device=device)
        env._v048_last_defender_touch_step = torch.full((n,), -100000, dtype=torch.long, device=device)
        env._v048_control_steps = torch.zeros(n, dtype=torch.long, device=device)
        env._v048_max_control_steps = torch.zeros(n, dtype=torch.long, device=device)
        env._v048_cross_contact_steps = torch.zeros(n, dtype=torch.long, device=device)
        env._v048_max_opponent_force = torch.zeros(n, device=device)
        env._v048_simultaneous_contact_steps = torch.zeros(
            n, dtype=torch.long, device=device
        )
        env._v048_upper_foul = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_lower_foul = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_foul = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_attacker_fell = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_defender_fell = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_fell = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_attacker_joint_violation = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        env._v048_defender_joint_violation = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        env._v048_joint_violation = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_attacker_torque_violation = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        env._v048_defender_torque_violation = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        env._v048_torque_violation = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_attacker_nonfinite = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_defender_nonfinite = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_nonfinite = torch.zeros(n, dtype=torch.bool, device=device)
        env._v048_attacker_max_abs_torque = torch.zeros(
            (n, ACTION_DIM), device=device
        )
        env._v048_defender_max_abs_torque = torch.zeros(
            (n, ACTION_DIM), device=device
        )
        env._v048_success = torch.zeros(n, dtype=torch.bool, device=device)


def reset_tackle_scene(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    evaluation_direction_rad: float | list[float] | torch.Tensor | None = None,
    evaluation_speed_mps: float | list[float] | torch.Tensor | None = None,
    evaluation_lateral_offset_m: float | list[float] | torch.Tensor | None = None,
) -> None:
    """Correlated reset with randomized attacker direction, speed and lateral offset."""

    _ensure_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    count = len(env_ids)
    def evaluation_values(
        value: float | list[float] | torch.Tensor, label: str
    ) -> torch.Tensor:
        values = torch.as_tensor(value, dtype=torch.float32, device=env.device)
        if values.ndim == 0:
            return values.repeat(count)
        if values.ndim != 1 or values.shape[0] != env.num_envs:
            raise ValueError(
                f"{label} must be scalar or one value per environment "
                f"({env.num_envs}), received {tuple(values.shape)}"
            )
        return values[env_ids]

    if evaluation_direction_rad is None:
        angle = (torch.rand(count, device=env.device) * 2.0 - 1.0) * math.pi
    else:
        angle = evaluation_values(evaluation_direction_rad, "evaluation_direction_rad")
    direction = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    lateral = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    if evaluation_speed_mps is None:
        speed = 0.15 + 0.50 * torch.rand(count, device=env.device)
    else:
        speed = evaluation_values(evaluation_speed_mps, "evaluation_speed_mps")
    if evaluation_lateral_offset_m is None:
        lateral_offset = (torch.rand(count, device=env.device) * 2.0 - 1.0) * 0.35
    else:
        lateral_offset = evaluation_values(
            evaluation_lateral_offset_m, "evaluation_lateral_offset_m"
        )

    attacker: Entity = env.scene["attacker"]
    defender: Entity = env.scene["defender"]
    ball: Entity = env.scene["ball"]
    origins = env.scene.env_origins[env_ids, :2]

    attacker_xy = origins - 0.85 * direction
    defender_xy = origins + 0.58 * direction + lateral_offset.unsqueeze(-1) * lateral
    ball_xy = attacker_xy + 0.12 * direction
    attacker_quat = quat_from_euler_xyz(torch.zeros_like(angle), torch.zeros_like(angle), angle)
    defender_yaw = angle + math.pi
    defender_quat = quat_from_euler_xyz(
        torch.zeros_like(angle), torch.zeros_like(angle), defender_yaw
    )

    attacker_state = attacker.data.default_root_state[env_ids].clone()
    defender_state = defender.data.default_root_state[env_ids].clone()
    ball_state = ball.data.default_root_state[env_ids].clone()
    attacker_state[:, :2] = attacker_xy
    defender_state[:, :2] = defender_xy
    ball_state[:, :2] = ball_xy
    attacker_state[:, 2] = 0.665
    defender_state[:, 2] = 0.665
    ball_state[:, 2] = 0.11
    attacker_state[:, 3:7] = attacker_quat
    defender_state[:, 3:7] = defender_quat
    ball_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    attacker_state[:, 7:9] = direction * speed.unsqueeze(-1)
    defender_state[:, 7:13] = 0.0
    ball_state[:, 7:9] = direction * speed.unsqueeze(-1)

    attacker.write_root_state_to_sim(attacker_state, env_ids)
    defender.write_root_state_to_sim(defender_state, env_ids)
    ball.write_root_state_to_sim(ball_state, env_ids)
    attacker.write_joint_state_to_sim(
        attacker.data.default_joint_pos[env_ids],
        attacker.data.default_joint_vel[env_ids],
        env_ids=env_ids,
    )
    defender.write_joint_state_to_sim(
        defender.data.default_joint_pos[env_ids],
        defender.data.default_joint_vel[env_ids],
        env_ids=env_ids,
    )

    env._v048_attack_direction[env_ids] = direction
    env._v048_attack_speed[env_ids] = speed
    for name in (
        "_v048_last_toucher",
        "_v048_first_touch_owner",
        "_v048_attacker_first_touch",
        "_v048_attacker_touched",
        "_v048_defender_touched",
        "_v048_turnover",
        "_v048_turnover_projection",
        "_v048_max_clearance",
        "_v048_clearance_delta",
        "_v048_control_steps",
        "_v048_max_control_steps",
        "_v048_cross_contact_steps",
        "_v048_max_opponent_force",
        "_v048_simultaneous_contact_steps",
        "_v048_upper_foul",
        "_v048_lower_foul",
        "_v048_foul",
        "_v048_attacker_fell",
        "_v048_defender_fell",
        "_v048_fell",
        "_v048_attacker_joint_violation",
        "_v048_defender_joint_violation",
        "_v048_joint_violation",
        "_v048_attacker_torque_violation",
        "_v048_defender_torque_violation",
        "_v048_torque_violation",
        "_v048_attacker_nonfinite",
        "_v048_defender_nonfinite",
        "_v048_nonfinite",
        "_v048_attacker_max_abs_torque",
        "_v048_defender_max_abs_torque",
        "_v048_success",
    ):
        getattr(env, name)[env_ids] = 0
    env._v048_last_defender_touch_step[env_ids] = -100000


def _sensor_contact(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    data = env.scene[name].data
    if data.force_history is not None:
        return (torch.linalg.vector_norm(data.force_history, dim=-1) > 1.0e-6).any(dim=(1, 2))
    assert data.found is not None
    return (data.found > 0).any(dim=-1)


def _sensor_max_force(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    data = env.scene[name].data
    force = data.force_history if data.force_history is not None else data.force
    assert force is not None
    return torch.linalg.vector_norm(force, dim=-1).flatten(1).amax(dim=-1)


def _upright(entity: Entity) -> torch.Tensor:
    return -entity.data.projected_gravity_b[:, 2]


def update_substep_joint_safety(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Accumulate all 23 actual joint torques on every 0.005 s physics step."""

    _ensure_state(env)
    lower = torch.as_tensor(T1_LOWER_LIMITS, device=env.device)
    upper = torch.as_tensor(T1_UPPER_LIMITS, device=env.device)
    limits = EFFORT_LIMITS.to(env.device)
    maximum_ratio = torch.zeros(env.num_envs, device=env.device)
    for prefix, entity_name in (("attacker", "attacker"), ("defender", "defender")):
        entity: Entity = env.scene[entity_name]
        torque = entity.data.qfrc_actuator
        joint_pos = entity.data.joint_pos
        joint_vel = entity.data.joint_vel
        if torque.shape[-1] != ACTION_DIM:
            raise RuntimeError(
                f"{entity_name} actual torque must contain {ACTION_DIM} joints, "
                f"received {torque.shape}"
            )
        max_torque = getattr(env, f"_v048_{prefix}_max_abs_torque")
        max_torque.copy_(torch.maximum(max_torque, torch.abs(torque)))
        joint_violation = (
            (joint_pos < lower - JOINT_LIMIT_TOLERANCE_RAD)
            | (joint_pos > upper + JOINT_LIMIT_TOLERANCE_RAD)
        ).any(dim=-1)
        torque_ratio = torch.abs(torque) / limits
        torque_violation = (torque_ratio > TORQUE_RATIO_TOLERANCE).any(dim=-1)
        nonfinite = ~(
            torch.isfinite(joint_pos).all(dim=-1)
            & torch.isfinite(joint_vel).all(dim=-1)
            & torch.isfinite(torque).all(dim=-1)
        )
        getattr(env, f"_v048_{prefix}_joint_violation").logical_or_(
            joint_violation
        )
        getattr(env, f"_v048_{prefix}_torque_violation").logical_or_(
            torque_violation
        )
        getattr(env, f"_v048_{prefix}_nonfinite").logical_or_(nonfinite)
        maximum_ratio = torch.maximum(maximum_ratio, torque_ratio.amax(dim=-1))
    env._v048_joint_violation = (
        env._v048_attacker_joint_violation | env._v048_defender_joint_violation
    )
    env._v048_torque_violation = (
        env._v048_attacker_torque_violation | env._v048_defender_torque_violation
    )
    env._v048_nonfinite |= (
        env._v048_attacker_nonfinite | env._v048_defender_nonfinite
    )
    return maximum_ratio


def update_tackle_tracker(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Update contact ownership and safety exactly once before gate termination."""

    _ensure_state(env)
    attacker: Entity = env.scene["attacker"]
    defender: Entity = env.scene["defender"]
    ball: Entity = env.scene["ball"]
    attacker_contact = _sensor_contact(env, "attacker_ball_legal")
    defender_contact = _sensor_contact(env, "defender_ball_legal")
    attacker_only = attacker_contact & ~defender_contact
    defender_only = defender_contact & ~attacker_contact
    no_first_touch = env._v048_first_touch_owner == 0
    attacker_is_first = no_first_touch & attacker_only
    defender_is_first = no_first_touch & defender_only
    simultaneous_is_first = no_first_touch & attacker_contact & defender_contact
    env._v048_first_touch_owner = torch.where(
        attacker_is_first,
        torch.ones_like(env._v048_first_touch_owner),
        torch.where(
            defender_is_first,
            torch.full_like(env._v048_first_touch_owner, 2),
            torch.where(
                simultaneous_is_first,
                torch.full_like(env._v048_first_touch_owner, 3),
                env._v048_first_touch_owner,
            ),
        ),
    )
    env._v048_attacker_first_touch |= attacker_is_first
    env._v048_simultaneous_contact_steps += (
        attacker_contact & defender_contact
    ).long()
    previous = env._v048_last_toucher.clone()
    env._v048_attacker_touched |= attacker_contact
    env._v048_defender_touched |= defender_contact
    new_turnover = (
        env._v048_attacker_first_touch
        & (previous == 1)
        & defender_only
        & ~env._v048_turnover
    )
    ball_projection = (ball.data.root_link_pos_w[:, :2] * env._v048_attack_direction).sum(dim=-1)
    env._v048_turnover_projection = torch.where(
        new_turnover, ball_projection, env._v048_turnover_projection
    )
    env._v048_turnover |= new_turnover
    env._v048_last_toucher = torch.where(
        attacker_only,
        torch.ones_like(env._v048_last_toucher),
        torch.where(defender_only, torch.full_like(env._v048_last_toucher, 2), env._v048_last_toucher),
    )
    env._v048_last_defender_touch_step = torch.where(
        defender_contact,
        torch.full_like(env._v048_last_defender_touch_step, env.common_step_counter),
        env._v048_last_defender_touch_step,
    )

    clearance = torch.clamp(env._v048_turnover_projection - ball_projection, min=0.0)
    previous_clearance = env._v048_max_clearance.clone()
    env._v048_max_clearance = torch.where(
        env._v048_turnover,
        torch.maximum(env._v048_max_clearance, clearance),
        env._v048_max_clearance,
    )
    env._v048_clearance_delta = env._v048_max_clearance - previous_clearance

    ball_distance = torch.linalg.vector_norm(
        ball.data.root_link_pos_w[:, :2] - defender.data.root_link_pos_w[:, :2], dim=-1
    )
    recent_touch = (env.common_step_counter - env._v048_last_defender_touch_step) * env.step_dt <= 0.15
    controlled = env._v048_turnover & (env._v048_last_toucher == 2) & recent_touch & (ball_distance <= 0.55)
    env._v048_control_steps = torch.where(
        controlled, env._v048_control_steps + 1, torch.zeros_like(env._v048_control_steps)
    )
    env._v048_max_control_steps = torch.maximum(
        env._v048_max_control_steps, env._v048_control_steps
    )

    cross_contact = _sensor_contact(env, "cross_robot_impact")
    upper_contact = _sensor_contact(env, "defender_upper_vs_attacker") | _sensor_contact(
        env, "attacker_upper_vs_defender"
    )
    lower_contact = _sensor_contact(env, "defender_lower_vs_attacker") | _sensor_contact(
        env, "attacker_lower_vs_defender"
    )
    env._v048_cross_contact_steps += cross_contact.long()
    env._v048_max_opponent_force = torch.maximum(
        env._v048_max_opponent_force, _sensor_max_force(env, "cross_robot_impact")
    )
    env._v048_upper_foul |= upper_contact
    env._v048_lower_foul |= lower_contact
    env._v048_foul |= upper_contact | lower_contact

    env._v048_attacker_fell |= (
        (attacker.data.root_link_pos_w[:, 2] < 0.34)
        | (_upright(attacker) < 0.20)
    )
    env._v048_defender_fell |= (
        (defender.data.root_link_pos_w[:, 2] < 0.34)
        | (_upright(defender) < 0.20)
    )
    env._v048_fell |= env._v048_attacker_fell | env._v048_defender_fell
    update_substep_joint_safety(env)
    for prefix, entity in (("attacker", attacker), ("defender", defender)):
        root_nonfinite = ~(
            torch.isfinite(entity.data.root_link_pose_w).all(dim=-1)
            & torch.isfinite(entity.data.root_link_vel_w).all(dim=-1)
        )
        getattr(env, f"_v048_{prefix}_nonfinite").logical_or_(root_nonfinite)
    env._v048_nonfinite |= (
        env._v048_attacker_nonfinite | env._v048_defender_nonfinite
    )
    env._v048_nonfinite |= ~(
        torch.isfinite(ball.data.root_link_pose_w).all(dim=-1)
        & torch.isfinite(ball.data.root_link_vel_w).all(dim=-1)
    )
    control_seconds = env._v048_max_control_steps.float() * env.step_dt
    env._v048_success = (
        env._v048_turnover
        & (env._v048_max_clearance >= MINIMUM_CLEARANCE_M)
        & (control_seconds >= MINIMUM_CONTROL_SECONDS)
        & ~env._v048_foul
        & ~env._v048_fell
        & ~env._v048_joint_violation
        & ~env._v048_torque_violation
        & ~env._v048_nonfinite
    )
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def safety_failure(env: ManagerBasedRlEnv) -> torch.Tensor:
    _ensure_state(env)
    return (
        env._v048_foul
        | env._v048_fell
        | env._v048_joint_violation
        | env._v048_torque_violation
        | env._v048_nonfinite
    )


def task_success(env: ManagerBasedRlEnv) -> torch.Tensor:
    _ensure_state(env)
    return env._v048_success


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length - 1


def defender_local_observation_components(
    env: ManagerBasedRlEnv,
) -> dict[str, torch.Tensor]:
    """Return named live tensors in the exact order used by the deployable actor."""

    _ensure_state(env)
    defender: Entity = env.scene["defender"]
    attacker: Entity = env.scene["attacker"]
    ball: Entity = env.scene["ball"]
    quat = defender.data.root_link_quat_w
    ball_rel = quat_apply_inverse(quat, ball.data.root_link_pos_w - defender.data.root_link_pos_w)
    ball_vel_rel = quat_apply_inverse(
        quat, ball.data.root_link_lin_vel_w - defender.data.root_link_lin_vel_w
    )
    attacker_rel = quat_apply_inverse(
        quat, attacker.data.root_link_pos_w - defender.data.root_link_pos_w
    )
    attacker_vel_rel = quat_apply_inverse(
        quat, attacker.data.root_link_lin_vel_w - defender.data.root_link_lin_vel_w
    )
    direction_3d = torch.cat(
        (env._v048_attack_direction, torch.zeros((env.num_envs, 1), device=env.device)), dim=-1
    )
    direction_b = quat_apply_inverse(quat, direction_3d)[:, :2]
    phase = (env.episode_length_buf.float() / max(1, env.max_episode_length)).unsqueeze(-1)
    # Shared v045 proprioception intentionally keeps the exact v045 units.
    # This makes the audited warm-start a true semantic column permutation;
    # football/opponent features remain bounded independently.
    return {
        "projected_gravity": defender.data.projected_gravity_b,
        "base_ang_vel": defender.data.root_link_ang_vel_b,
        "base_lin_vel": defender.data.root_link_lin_vel_b,
        "joint_pos": defender.data.joint_pos - defender.data.default_joint_pos,
        "joint_vel": defender.data.joint_vel,
        "last_action": env.action_manager.action,
        "ball_relative_position": ball_rel * 0.5,
        "ball_relative_velocity": ball_vel_rel * 0.5,
        "attacker_relative_position": attacker_rel * 0.5,
        "attacker_relative_velocity": attacker_vel_rel * 0.5,
        "attack_direction": direction_b,
        "episode_phase": phase,
    }


def defender_local_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Deployable local observation: IMU, encoders, action history and local vision."""

    components = defender_local_observation_components(env)
    expected_names = [name for name, _ in V048_ACTOR_OBSERVATION_LAYOUT]
    if list(components) != expected_names:
        raise RuntimeError(
            f"v048 actor observation order drifted: {list(components)} != {expected_names}"
        )
    for name, (start, stop) in V048_ACTOR_OBSERVATION_LAYOUT:
        if components[name].shape[1] != stop - start:
            raise RuntimeError(
                f"v048 observation component {name} expected {stop-start}, "
                f"got {components[name].shape}"
            )
    observation = torch.cat(tuple(components.values()), dim=-1)
    if observation.shape[1] != ACTOR_OBSERVATION_DIM:
        raise RuntimeError(f"v048 actor observation must be {ACTOR_OBSERVATION_DIM}, got {observation.shape}")
    return observation


def privileged_critic_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Training-only world state, contacts, ownership, clearance and safety."""

    _ensure_state(env)
    attacker: Entity = env.scene["attacker"]
    defender: Entity = env.scene["defender"]
    ball: Entity = env.scene["ball"]
    state = torch.stack(
        (
            env._v048_attacker_first_touch.float(),
            env._v048_defender_touched.float(),
            env._v048_turnover.float(),
            env._v048_last_toucher.float() / 2.0,
            env._v048_max_clearance,
            env._v048_max_control_steps.float() * env.step_dt,
            env._v048_cross_contact_steps.float() * env.step_dt,
            env._v048_max_opponent_force / 500.0,
            env._v048_foul.float(),
            env._v048_fell.float(),
            env._v048_joint_violation.float(),
            env._v048_torque_violation.float(),
        ),
        dim=-1,
    )
    return torch.cat(
        (
            attacker.data.root_link_pose_w,
            attacker.data.root_link_vel_w,
            attacker.data.joint_pos,
            attacker.data.joint_vel,
            defender.data.root_link_pose_w,
            defender.data.root_link_vel_w,
            ball.data.root_link_pose_w,
            ball.data.root_link_vel_w,
            state,
        ),
        dim=-1,
    )


def distance_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    defender: Entity = env.scene["defender"]
    ball: Entity = env.scene["ball"]
    distance = torch.linalg.vector_norm(
        defender.data.root_link_pos_w[:, :2] - ball.data.root_link_pos_w[:, :2], dim=-1
    )
    return torch.exp(-2.0 * distance)


def turnover_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    _ensure_state(env)
    return env._v048_turnover.float()


def clearance_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    _ensure_state(env)
    return env._v048_clearance_delta


def control_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    _ensure_state(env)
    return (env._v048_control_steps > 0).float()


def upright_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.clamp(_upright(env.scene["defender"]), min=0.0, max=1.0)


def action_rate_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.square(env.action_manager.action - env.action_manager.prev_action).sum(dim=-1)


def torque_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    defender: Entity = env.scene["defender"]
    return torch.square(defender.data.qfrc_actuator / EFFORT_LIMITS.to(env.device)).sum(dim=-1)


def foul_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    _ensure_state(env)
    return env._v048_foul.float()


def fall_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    _ensure_state(env)
    return env._v048_fell.float()


def exact_rcss_tackle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    defender_clip = {
        name: (float(lower), float(upper))
        for name, lower, upper in zip(RCSS_JOINT_NAMES, T1_LOWER_LIMITS, T1_UPPER_LIMITS, strict=True)
    }
    sensors = (
        ContactSensorCfg(
            name="attacker_ball_legal",
            primary=ContactMatch(mode="geom", pattern=r"(left|right)_(foot|calf)", entity="attacker"),
            secondary=ContactMatch(mode="geom", pattern="ball", entity="ball"),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=4,
        ),
        ContactSensorCfg(
            name="defender_lower_vs_attacker",
            primary=ContactMatch(
                mode="geom",
                pattern=r"(left|right)_(thigh|calf|foot)",
                entity="defender",
            ),
            secondary=ContactMatch(mode="subtree", pattern="torso", entity="attacker"),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=4,
        ),
        ContactSensorCfg(
            name="attacker_lower_vs_defender",
            primary=ContactMatch(
                mode="geom",
                pattern=r"(left|right)_(thigh|calf|foot)",
                entity="attacker",
            ),
            secondary=ContactMatch(mode="subtree", pattern="torso", entity="defender"),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=4,
        ),
        ContactSensorCfg(
            name="defender_ball_legal",
            primary=ContactMatch(mode="geom", pattern=r"(left|right)_(foot|calf)", entity="defender"),
            secondary=ContactMatch(mode="geom", pattern="ball", entity="ball"),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=4,
        ),
        ContactSensorCfg(
            name="cross_robot_impact",
            primary=ContactMatch(mode="subtree", pattern="torso", entity="defender"),
            secondary=ContactMatch(mode="subtree", pattern="torso", entity="attacker"),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=4,
            history_length=4,
        ),
        ContactSensorCfg(
            name="defender_upper_vs_attacker",
            primary=ContactMatch(
                mode="geom",
                pattern=r"(torso|head|left_forearm|right_forearm|left_hand|right_hand)",
                entity="defender",
            ),
            secondary=ContactMatch(mode="subtree", pattern="torso", entity="attacker"),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=4,
        ),
        ContactSensorCfg(
            name="attacker_upper_vs_defender",
            primary=ContactMatch(
                mode="geom",
                pattern=r"(torso|head|left_forearm|right_forearm|left_hand|right_hand)",
                entity="attacker",
            ),
            secondary=ContactMatch(mode="subtree", pattern="torso", entity="defender"),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=4,
        ),
    )
    actor_terms = {
        "local_deployable_state": ObservationTermCfg(func=defender_local_observation),
    }
    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
        "critic": ObservationGroupCfg(
            {
                "local_deployable_state": ObservationTermCfg(func=defender_local_observation),
                "privileged_world_state": ObservationTermCfg(func=privileged_critic_observation),
            },
            enable_corruption=False,
        ),
    }
    actions: dict[str, ActionTermCfg] = {
        "defender_fullbody": V048SafeT1JointPositionActionCfg(
            entity_name="defender",
            actuator_names=RCSS_JOINT_NAMES,
            preserve_order=True,
            scale=0.30,
            use_default_offset=True,
            clip=defender_clip,
        ),
        "attacker_physics_teacher": ScriptedAttackerActionCfg(entity_name="attacker"),
    }
    events = {
        "correlated_tackle_reset": EventTermCfg(func=reset_tackle_scene, mode="reset"),
    }
    terminations = {
        "tracker_update": TerminationTermCfg(func=update_tackle_tracker),
        "safety_failure": TerminationTermCfg(func=safety_failure),
        "task_success": TerminationTermCfg(func=task_success),
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }
    rewards = {
        "approach_ball": RewardTermCfg(func=distance_reward, weight=1.0),
        "turnover": RewardTermCfg(func=turnover_reward, weight=4.0),
        "clearance_progress": RewardTermCfg(func=clearance_reward, weight=20.0),
        "post_tackle_control": RewardTermCfg(func=control_reward, weight=2.0),
        "upright": RewardTermCfg(func=upright_reward, weight=0.5),
        "action_rate": RewardTermCfg(func=action_rate_penalty, weight=-0.01),
        "torque": RewardTermCfg(func=torque_penalty, weight=-0.002),
        "foul": RewardTermCfg(func=foul_penalty, weight=-30.0),
        "fall": RewardTermCfg(func=fall_penalty, weight=-30.0),
    }
    metrics = {
        "maximum_actual_torque_ratio_per_physics_substep": MetricsTermCfg(
            func=update_substep_joint_safety,
            per_substep=True,
        ),
    }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={
                "attacker": get_exact_rcss_t1_cfg(),
                "defender": get_exact_rcss_t1_cfg(),
                "ball": EntityCfg(spec_fn=get_rcss_ball_spec, init_state=BALL_STATE),
            },
            sensors=sensors,
            spec_fn=_add_exact_pitch,
            num_envs=1,
            env_spacing=4.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        metrics=metrics,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="defender",
            body_name="torso",
            distance=4.5,
            elevation=-20.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.005),
            nconmax=512,
            njmax=2048,
        ),
        decimation=4,
        episode_length_s=8.0,
        scale_rewards_by_dt=False,
        seed=48000,
    )
    if play:
        cfg.observations["actor"].enable_corruption = False
    return cfg


def v048_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.6,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="v048_fullbody_legal_tackle",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=3000,
        logger="tensorboard",
    )


def register_v048_task() -> None:
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=exact_rcss_tackle_env_cfg(),
        play_env_cfg=exact_rcss_tackle_env_cfg(play=True),
        rl_cfg=v048_ppo_runner_cfg(),
    )
