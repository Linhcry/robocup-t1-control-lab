"""Per-environment ball placement and directional-pass command sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


def _sensor_found(sensor: ContactSensor) -> torch.Tensor:
    found = sensor.data.found
    if found is None:
        raise RuntimeError(f"contact sensor {sensor.cfg.name!r} has no found field")
    instantaneous = (found > 0.0).any(dim=-1)
    force_history = sensor.data.force_history
    if force_history is None:
        raise RuntimeError(
            f"contact sensor {sensor.cfg.name!r} has no substep force history"
        )
    # [environment, primary channel, physics-history, xyz].  Reading all four
    # physics substeps prevents a short foot strike (or illegal body touch)
    # between two 50 Hz policy observations from disappearing.
    historical = (torch.linalg.vector_norm(force_history, dim=-1) > 0.0).any(
        dim=(-1, -2)
    )
    return instantaneous | historical


class DirectionalPassCommand(CommandTerm):
    """Sample ball/target geometry and latch real MuJoCo contact events."""

    cfg: "DirectionalPassCommandCfg"

    def __init__(self, cfg: "DirectionalPassCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.ball: Entity = env.scene[cfg.ball_entity_name]
        self.foot_sensor: ContactSensor = env.scene[cfg.foot_sensor_name]
        self.upper_sensor: ContactSensor = env.scene[cfg.upper_sensor_name]
        self.direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.initial_ball_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.pass_distance = torch.zeros(self.num_envs, device=self.device)
        self.foot_touched = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.upper_body_touched = torch.zeros_like(self.foot_touched)
        self.episode_success = torch.zeros_like(self.foot_touched)
        self.progress_m = torch.zeros(self.num_envs, device=self.device)
        self.lateral_error_m = torch.zeros(self.num_envs, device=self.device)
        self.metrics.update(
            {
                "true_foot_contact": torch.zeros(self.num_envs, device=self.device),
                "upper_body_ball_contact": torch.zeros(
                    self.num_envs, device=self.device
                ),
                "forward_progress_m": torch.zeros(self.num_envs, device=self.device),
                "lateral_error_m": torch.zeros(self.num_envs, device=self.device),
                "pass_success": torch.zeros(self.num_envs, device=self.device),
            }
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [self.direction_w[:, :2], self.pass_distance[:, None]], dim=-1
        )

    def current_foot_contact(self) -> torch.Tensor:
        return _sensor_found(self.foot_sensor)

    def current_upper_contact(self) -> torch.Tensor:
        return _sensor_found(self.upper_sensor)

    def true_foot_contact_latched(self) -> torch.Tensor:
        return self.foot_touched | self.current_foot_contact()

    def failure_contact_latched(self) -> torch.Tensor:
        return self.upper_body_touched | self.current_upper_contact()

    def success_now(self) -> torch.Tensor:
        progress_ok = self.progress_m >= (
            self.pass_distance * self.cfg.success_distance_fraction
        )
        lateral_ok = self.lateral_error_m <= self.cfg.success_corridor_half_width_m
        height_ok = self.ball.data.root_link_pos_w[:, 2] <= self.cfg.maximum_success_height_m
        return (
            self.true_foot_contact_latched()
            & ~self.failure_contact_latched()
            & progress_ok
            & lateral_ok
            & height_ok
        )

    def _update_metrics(self) -> None:
        foot_now = self.current_foot_contact()
        upper_now = self.current_upper_contact()
        self.foot_touched |= foot_now
        self.upper_body_touched |= upper_now

        displacement = self.ball.data.root_link_pos_w - self.initial_ball_pos_w
        self.progress_m = torch.sum(displacement * self.direction_w, dim=-1)
        lateral = displacement[:, :2] - (
            self.progress_m[:, None] * self.direction_w[:, :2]
        )
        self.lateral_error_m = torch.linalg.vector_norm(lateral, dim=-1)
        self.episode_success |= self.success_now()

        self.metrics["true_foot_contact"] = self.foot_touched.float()
        self.metrics["upper_body_ball_contact"] = self.upper_body_touched.float()
        self.metrics["forward_progress_m"] = self.progress_m
        self.metrics["lateral_error_m"] = self.lateral_error_m
        self.metrics["pass_success"] = self.episode_success.float()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self.foot_touched[env_ids] = False
        self.upper_body_touched[env_ids] = False
        self.episode_success[env_ids] = False
        self.progress_m[env_ids] = 0.0
        self.lateral_error_m[env_ids] = 0.0

        angle = sample_uniform(
            self.cfg.target_angle_range_rad[0],
            self.cfg.target_angle_range_rad[1],
            (n,),
            device=self.device,
        )
        direction = torch.stack(
            [torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], dim=-1
        )
        distance = sample_uniform(
            self.cfg.pass_distance_range_m[0],
            self.cfg.pass_distance_range_m[1],
            (n,),
            device=self.device,
        )
        ball_x = sample_uniform(
            self.cfg.ball_x_range_m[0],
            self.cfg.ball_x_range_m[1],
            (n,),
            device=self.device,
        )
        ball_y = sample_uniform(
            self.cfg.ball_y_range_m[0],
            self.cfg.ball_y_range_m[1],
            (n,),
            device=self.device,
        )
        ball_pos = self._env.scene.env_origins[env_ids].clone()
        ball_pos[:, 0] += ball_x
        ball_pos[:, 1] += ball_y
        ball_pos[:, 2] += self.cfg.ball_radius_m + self.cfg.ball_reset_clearance_m

        self.direction_w[env_ids] = direction
        self.pass_distance[env_ids] = distance
        self.initial_ball_pos_w[env_ids] = ball_pos
        self.target_pos_w[env_ids] = ball_pos + direction * distance[:, None]

        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0
        pose = torch.cat([ball_pos, quat], dim=-1)
        velocity = torch.zeros(n, 6, device=self.device)
        self.ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.ball.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        for batch in visualizer.get_env_indices(self.num_envs):
            visualizer.add_sphere(
                center=self.target_pos_w[batch].cpu().numpy(),
                radius=0.06,
                color=(0.1, 0.9, 0.2, 0.45),
                label=f"v046_pass_target_{batch}",
            )


@dataclass(kw_only=True)
class DirectionalPassCommandCfg(CommandTermCfg):
    """Randomization and strict success contract for stage-one passing."""

    ball_entity_name: str = "ball"
    foot_sensor_name: str = "foot_ball_contact"
    upper_sensor_name: str = "upper_body_ball_contact"
    ball_radius_m: float = 0.11
    ball_reset_clearance_m: float = 0.002
    ball_x_range_m: tuple[float, float] = (0.28, 0.48)
    ball_y_range_m: tuple[float, float] = (-0.16, 0.16)
    target_angle_range_rad: tuple[float, float] = (-math.pi / 3, math.pi / 3)
    pass_distance_range_m: tuple[float, float] = (1.0, 3.0)
    success_distance_fraction: float = 0.85
    success_corridor_half_width_m: float = 0.35
    maximum_success_height_m: float = 0.35

    def build(self, env: "ManagerBasedRlEnv") -> DirectionalPassCommand:
        return DirectionalPassCommand(self, env)
