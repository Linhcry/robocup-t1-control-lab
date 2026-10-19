"""Staged pass rewards, all success terms gated by a real foot contact."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from experiments.v046_fullbody_pass.mdp.commands import DirectionalPassCommand
from locomotion.rcss_t1_controller import TORQUE_LIMIT
from mjlab.entity import Entity

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# Training, strict evaluation and the competition controller use one identical
# conservative 23-joint safety contract.  The RCSS XML's mechanical caps are
# not a permission to train above the limits used by the deployed controller.
EFFORT_LIMITS = tuple(float(value) for value in TORQUE_LIMIT)


def _state(env: "ManagerBasedRlEnv") -> tuple[Entity, Entity, DirectionalPassCommand]:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    command = cast(
        DirectionalPassCommand, env.command_manager.get_term("directional_pass")
    )
    return robot, ball, command


def posture_reward(env: "ManagerBasedRlEnv", height: float = 0.665) -> torch.Tensor:
    robot, _, _ = _state(env)
    gravity_error = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=-1)
    height_error = torch.square(robot.data.root_link_pos_w[:, 2] - height)
    return torch.exp(-gravity_error / 0.08) * torch.exp(-height_error / 0.04)


def approach_ball_reward(env: "ManagerBasedRlEnv", desired_m: float = 0.24) -> torch.Tensor:
    """Stage-one shaping: move into a controllable stance behind the ball."""

    robot, ball, command = _state(env)
    delta = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    distance_error = torch.abs(torch.linalg.vector_norm(delta, dim=-1) - desired_m)
    behind_point = ball.data.root_link_pos_w[:, :2] - command.direction_w[:, :2] * desired_m
    behind_error = torch.linalg.vector_norm(
        robot.data.root_link_pos_w[:, :2] - behind_point, dim=-1
    )
    before_contact = (~command.true_foot_contact_latched()).float()
    return before_contact * torch.exp(-distance_error / 0.25) * torch.exp(
        -behind_error / 0.6
    )


def foot_contact_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Nonzero only for actual named foot-geom/ball contacts."""

    _, _, command = _state(env)
    return command.current_foot_contact().float()


def ball_forward_speed_reward(
    env: "ManagerBasedRlEnv", maximum_speed_mps: float = 5.0
) -> torch.Tensor:
    _, ball, command = _state(env)
    forward_speed = torch.sum(
        ball.data.root_link_lin_vel_w * command.direction_w, dim=-1
    ).clamp(min=0.0, max=maximum_speed_mps)
    return command.true_foot_contact_latched().float() * (
        forward_speed / maximum_speed_mps
    )


def ball_direction_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    _, ball, command = _state(env)
    velocity = ball.data.root_link_lin_vel_w[:, :2]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    cosine = torch.sum(velocity * command.direction_w[:, :2], dim=-1) / speed.clamp_min(
        1.0e-4
    )
    return (
        command.true_foot_contact_latched().float()
        * cosine.clamp(min=-1.0, max=1.0)
        * (speed > 0.1).float()
    )


def ball_progress_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    _, _, command = _state(env)
    normalized = (command.progress_m / command.pass_distance.clamp_min(0.1)).clamp(
        min=0.0, max=1.0
    )
    return command.true_foot_contact_latched().float() * normalized


def target_corridor_reward(env: "ManagerBasedRlEnv", std_m: float = 0.25) -> torch.Tensor:
    _, _, command = _state(env)
    return command.true_foot_contact_latched().float() * torch.exp(
        -torch.square(command.lateral_error_m / std_m)
    )


def pass_success_reward(env: "ManagerBasedRlEnv") -> torch.Tensor:
    _, _, command = _state(env)
    return command.success_now().float()


def upper_body_ball_penalty(env: "ManagerBasedRlEnv") -> torch.Tensor:
    _, _, command = _state(env)
    return command.current_upper_contact().float()


def fall_penalty(env: "ManagerBasedRlEnv", minimum_height_m: float = 0.48) -> torch.Tensor:
    robot, _, _ = _state(env)
    tilt = torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    return ((robot.data.root_link_pos_w[:, 2] < minimum_height_m) | (tilt > 0.72)).float()


def torque_usage_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    robot, _, _ = _state(env)
    limits = torch.tensor(EFFORT_LIMITS, device=env.device).view(1, 23)
    normalized = robot.data.qfrc_actuator / limits
    return torch.sum(torch.square(normalized), dim=-1)
