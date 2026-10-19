"""Deployment-aligned bounded joint-position actions for v045.

The stock MJLab relative-position action recomputes ``q + delta`` inside every
physics substep.  RCSS deployment instead computes one target at the 50 Hz
policy boundary and holds that target for the four 5 ms physics substeps.  This
term deliberately implements the latter contract and exposes read-only
telemetry for the persistent stability watchdog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions.actions import (
    RelativeJointPositionAction,
    RelativeJointPositionActionCfg,
)
from locomotion.t1_spec import (
    T1_ACTION_DELTA_LIMIT_RAD,
    T1_LOWER_LIMITS,
    T1_TARGET_LIMIT_MARGIN_RAD,
    T1_UPPER_LIMITS,
    T1_VELOCITY_BRAKE_LOOKAHEAD_S,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


ACTION_DELTA_LIMIT_RAD = T1_ACTION_DELTA_LIMIT_RAD
TARGET_LIMIT_MARGIN_RAD = T1_TARGET_LIMIT_MARGIN_RAD
VELOCITY_BRAKE_LOOKAHEAD_S = T1_VELOCITY_BRAKE_LOOKAHEAD_S


@dataclass(kw_only=True)
class RcssSafeRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Build a relative action that clamps the final target to hard limits."""

    def build(self, env: ManagerBasedRlEnv) -> RcssSafeRelativeJointPositionAction:
        return RcssSafeRelativeJointPositionAction(cfg=self, env=env)


class RcssSafeRelativeJointPositionAction(RelativeJointPositionAction):
    """Match the formal RCSS controller's action and target safety semantics."""

    def __init__(self, cfg: RcssSafeRelativeJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        if self.action_dim != 23:
            raise ValueError(f"v045 requires exactly 23 actions, got {self.action_dim}")
        canonical_lower = torch.as_tensor(
            T1_LOWER_LIMITS, device=self.device, dtype=self._raw_actions.dtype
        ).unsqueeze(0)
        canonical_upper = torch.as_tensor(
            T1_UPPER_LIMITS, device=self.device, dtype=self._raw_actions.dtype
        ).unsqueeze(0)
        model_limits = self._entity.data.joint_pos_limits[:, self._target_ids]
        self._deployment_lower = torch.maximum(
            model_limits[..., 0], canonical_lower
        )
        self._deployment_upper = torch.minimum(
            model_limits[..., 1], canonical_upper
        )
        self._safe_target_lower = self._deployment_lower + TARGET_LIMIT_MARGIN_RAD
        self._safe_target_upper = self._deployment_upper - TARGET_LIMIT_MARGIN_RAD
        self._safe_target_midpoint = 0.5 * (
            self._safe_target_lower + self._safe_target_upper
        )
        if not bool((self._safe_target_lower < self._safe_target_upper).all()):
            raise ValueError("v045 target safety margin collapses a joint range")
        self._bounded_delta = torch.zeros_like(self._processed_actions)
        self._final_target = self._entity.data.joint_pos[
            :, self._target_ids
        ].clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        """Convert one 50 Hz actor output into one target held for four substeps."""

        super().process_actions(actions)
        self._bounded_delta = torch.clamp(
            self._processed_actions,
            min=-ACTION_DELTA_LIMIT_RAD,
            max=ACTION_DELTA_LIMIT_RAD,
        )
        current_pos = self._entity.data.joint_pos[:, self._target_ids]
        current_vel = self._entity.data.joint_vel[:, self._target_ids]
        requested = current_pos + self._bounded_delta
        dynamic_lower = torch.minimum(
            self._safe_target_lower
            + torch.relu(-current_vel) * VELOCITY_BRAKE_LOOKAHEAD_S,
            self._safe_target_midpoint,
        )
        dynamic_upper = torch.maximum(
            self._safe_target_upper
            - torch.relu(current_vel) * VELOCITY_BRAKE_LOOKAHEAD_S,
            self._safe_target_midpoint,
        )
        self._final_target = torch.maximum(
            torch.minimum(requested, dynamic_upper),
            dynamic_lower,
        )
        if not torch.isfinite(self._final_target).all():
            raise FloatingPointError("v045 produced a non-finite joint target")

    def apply_actions(self) -> None:
        """Hold the already-processed 50 Hz target during every physics substep."""

        self._entity.set_joint_position_target(
            self._final_target, joint_ids=self._target_ids
        )

    @property
    def processed_delta(self) -> torch.Tensor:
        """Scaled 23D relative delta used to construct the held target."""

        return self._bounded_delta

    @property
    def unbounded_scaled_delta(self) -> torch.Tensor:
        """Raw actor output after the official 0.6 scale, before the rate guard."""

        return self._processed_actions

    @property
    def raw_action(self) -> torch.Tensor:
        """Unclipped 23D actor output received at the 50 Hz policy boundary."""

        return self._raw_actions

    @property
    def final_target(self) -> torch.Tensor:
        """Deployment-clamped absolute 23D target held across decimation."""

        return self._final_target

    @property
    def deployment_lower_limits(self) -> torch.Tensor:
        return self._deployment_lower

    @property
    def deployment_upper_limits(self) -> torch.Tensor:
        return self._deployment_upper
