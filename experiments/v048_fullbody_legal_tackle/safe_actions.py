"""Fail-closed, GPU-vectorized 23-DoF RCSS T1 position actions for v048."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import (
    JointPositionAction,
    JointPositionActionCfg,
)

from locomotion.t1_spec import (
    T1_JOINT_NAMES,
    T1_LOWER_LIMITS,
    T1_UPPER_LIMITS,
    clamp_t1_targets,
)


ACTION_DIM = 23


def _canonical_name(name: str) -> str:
    return "Head_yaw" if name == "AAHead_yaw" else name


class V048SafeT1JointPositionAction(JointPositionAction):
    """Reject invalid actions and clamp every final target to canonical limits.

    The public NumPy ``clamp_t1_targets`` function is the project's single-
    robot deployment contract.  It is used here to derive and verify the exact
    canonical bounds.  The same bounds are then applied with Torch so thousands
    of MJLab environments remain GPU-vectorized.
    """

    cfg: "V048SafeT1JointPositionActionCfg"

    def __init__(self, cfg: "V048SafeT1JointPositionActionCfg", env):
        super().__init__(cfg=cfg, env=env)
        names = tuple(_canonical_name(name) for name in self.target_names)
        if self.action_dim != ACTION_DIM or names != tuple(T1_JOINT_NAMES):
            raise ValueError(
                "v048 action order must be the canonical 23-DoF T1 order; "
                f"received {self.action_dim} targets: {names}"
            )
        canonical_lower = clamp_t1_targets(T1_LOWER_LIMITS.copy())
        canonical_upper = clamp_t1_targets(T1_UPPER_LIMITS.copy())
        xml_limits = self._entity.data.joint_pos_limits[:, self._target_ids]
        lower = torch.as_tensor(
            canonical_lower, dtype=torch.float32, device=self.device
        ).view(1, ACTION_DIM)
        upper = torch.as_tensor(
            canonical_upper, dtype=torch.float32, device=self.device
        ).view(1, ACTION_DIM)
        self._lower = torch.maximum(lower, xml_limits[..., 0])
        self._upper = torch.minimum(upper, xml_limits[..., 1])
        if not bool(torch.all(self._lower < self._upper)):
            raise ValueError("canonical and installed RCSS T1 limits do not overlap")

    @property
    def clamped_target(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.ndim != 2 or actions.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(
                f"v048 raw action must be ({self.num_envs}, {ACTION_DIM}), "
                f"received {tuple(actions.shape)}"
            )
        if not bool(torch.isfinite(actions).all()):
            raise FloatingPointError("v048 refuses NaN/Inf policy actions")
        super().process_actions(torch.clamp(actions, -1.0, 1.0))
        self._processed_actions = torch.maximum(
            self._lower, torch.minimum(self._upper, self._processed_actions)
        )

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = self._processed_actions - encoder_bias
        target = torch.maximum(self._lower, torch.minimum(self._upper, target))
        if not bool(torch.isfinite(target).all()):
            raise FloatingPointError("v048 produced a non-finite joint target")
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class V048SafeT1JointPositionActionCfg(JointPositionActionCfg):
    def build(self, env) -> V048SafeT1JointPositionAction:
        if not self.use_default_offset:
            raise ValueError("v048 requires the audited default-pose action offset")
        return V048SafeT1JointPositionAction(self, env)
