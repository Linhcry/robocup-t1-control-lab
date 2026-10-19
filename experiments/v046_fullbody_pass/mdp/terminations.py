"""Strict v046 safety and pass-outcome terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from experiments.v046_fullbody_pass.mdp.commands import DirectionalPassCommand
from experiments.v046_fullbody_pass.mdp.rewards import EFFORT_LIMITS
from mjlab.entity import Entity

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _state(env: "ManagerBasedRlEnv") -> tuple[Entity, Entity, DirectionalPassCommand]:
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    command = cast(
        DirectionalPassCommand, env.command_manager.get_term("directional_pass")
    )
    return robot, ball, command


def upper_body_ball_contact(env: "ManagerBasedRlEnv") -> torch.Tensor:
    _, _, command = _state(env)
    return command.current_upper_contact()


def fallen(
    env: "ManagerBasedRlEnv",
    minimum_height_m: float = 0.48,
    maximum_tilt_sine: float = 0.72,
) -> torch.Tensor:
    robot, _, _ = _state(env)
    tilt = torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=-1)
    return (robot.data.root_link_pos_w[:, 2] < minimum_height_m) | (
        tilt > maximum_tilt_sine
    )


def hard_joint_limit_violation(
    env: "ManagerBasedRlEnv", tolerance_rad: float = 1.0e-4
) -> torch.Tensor:
    robot, _, _ = _state(env)
    limits = robot.data.default_joint_pos_limits
    q = robot.data.joint_pos
    return ((q < limits[..., 0] - tolerance_rad) | (q > limits[..., 1] + tolerance_rad)).any(
        dim=-1
    )


def hard_torque_limit_violation(
    env: "ManagerBasedRlEnv", tolerance_fraction: float = 1.0e-3
) -> torch.Tensor:
    robot, _, _ = _state(env)
    limits = torch.tensor(EFFORT_LIMITS, device=env.device).view(1, 23)
    return (robot.data.qfrc_actuator.abs() > limits * (1.0 + tolerance_fraction)).any(
        dim=-1
    )


def ball_out_of_bounds(
    env: "ManagerBasedRlEnv", maximum_distance_m: float = 5.0
) -> torch.Tensor:
    _, ball, _ = _state(env)
    local = ball.data.root_link_pos_w - env.scene.env_origins
    return (
        (local[:, :2].abs() > maximum_distance_m).any(dim=-1)
        | (local[:, 2] < -0.05)
        | (local[:, 2] > 1.5)
    )


def pass_success(env: "ManagerBasedRlEnv") -> torch.Tensor:
    _, _, command = _state(env)
    return command.success_now()

