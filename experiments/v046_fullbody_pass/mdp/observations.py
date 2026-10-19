"""Deployable actor observations and explicitly privileged critic terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from experiments.v046_fullbody_pass.mdp.commands import DirectionalPassCommand
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _entities(env: "ManagerBasedRlEnv") -> tuple[Entity, Entity, DirectionalPassCommand]:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    command = cast(
        DirectionalPassCommand, env.command_manager.get_term("directional_pass")
    )
    return robot, ball, command


def ball_relative_position_b(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Perceived 3-D vector from torso to ball in the torso frame."""

    robot, ball, _ = _entities(env)
    delta_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    return quat_apply(quat_inv(robot.data.root_link_quat_w), delta_w)


def ball_relative_velocity_b(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Estimated ball velocity relative to the torso, in the torso frame."""

    robot, ball, _ = _entities(env)
    delta_w = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
    return quat_apply(quat_inv(robot.data.root_link_quat_w), delta_w)


def target_direction_b(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Unit target direction in the torso horizontal frame."""

    robot, _, command = _entities(env)
    direction_b = quat_apply(quat_inv(robot.data.root_link_quat_w), command.direction_w)
    horizontal = direction_b[:, :2]
    return horizontal / torch.linalg.vector_norm(
        horizontal, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)


def target_distance(env: "ManagerBasedRlEnv", scale_m: float = 3.0) -> torch.Tensor:
    """Requested pass distance, normalized only by a public task constant."""

    _, _, command = _entities(env)
    return (command.pass_distance / scale_m).unsqueeze(-1)


def privileged_ball_world_state(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Exact simulator ball pose/velocity; critic-only."""

    _, ball, _ = _entities(env)
    pos_local = ball.data.root_link_pos_w - env.scene.env_origins
    return torch.cat([pos_local, ball.data.root_link_vel_w], dim=-1)


def contact_state_privileged(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Exact latched contact state; critic-only and never exported to actor."""

    _, _, command = _entities(env)
    return torch.stack(
        [command.foot_touched.float(), command.upper_body_touched.float()], dim=-1
    )


def pass_progress_privileged(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Exact projected progress and lateral error; critic-only."""

    _, _, command = _entities(env)
    return torch.stack([command.progress_m, command.lateral_error_m], dim=-1)


def root_height_privileged(env: "ManagerBasedRlEnv") -> torch.Tensor:
    robot, _, _ = _entities(env)
    return robot.data.root_link_pos_w[:, 2:3]

