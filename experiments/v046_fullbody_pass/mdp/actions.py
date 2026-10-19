"""Fail-closed, GPU-vectorized 23-DoF T1 position action."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from locomotion.t1_spec import T1_JOINT_NAMES, T1_LOWER_LIMITS, T1_UPPER_LIMITS
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg


def _canonical_name(name: str) -> str:
    return "Head_yaw" if name == "AAHead_yaw" else name


class SafeT1JointPositionAction(BaseAction):
    """Scale policy actions, reject invalid tensors and clamp every target.

    ``locomotion.t1_spec.clamp_t1_targets`` is the scalar NumPy deployment
    contract.  Moving a 4096-environment action batch through NumPy would break
    GPU parallelism, so this term performs the exact same min/max operation with
    tensors copied from that contract.  The semantic audit compares both paths
    on boundary and non-finite-free fixtures.
    """

    cfg: "SafeT1JointPositionActionCfg"

    def __init__(self, cfg: "SafeT1JointPositionActionCfg", env):
        super().__init__(cfg=cfg, env=env)
        names = tuple(_canonical_name(name) for name in self._target_names)
        if names != tuple(T1_JOINT_NAMES):
            raise ValueError(
                "v046 action order must be the canonical 23-DoF T1 order; "
                f"received {names}"
            )
        if self.action_dim != 23:
            raise ValueError(f"v046 requires 23 actions, received {self.action_dim}")
        self._lower = torch.as_tensor(
            T1_LOWER_LIMITS, device=self.device, dtype=torch.float32
        ).view(1, 23)
        self._upper = torch.as_tensor(
            T1_UPPER_LIMITS, device=self.device, dtype=torch.float32
        ).view(1, 23)
        self._offset = self._entity.data.default_joint_pos[:, self._target_ids].clone()

    @property
    def clamped_target(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.ndim != 2 or actions.shape != (self.num_envs, 23):
            raise ValueError(
                "T1 raw action must have shape "
                f"({self.num_envs}, 23), received {tuple(actions.shape)}"
            )
        if not torch.isfinite(actions).all():
            raise FloatingPointError("v046 rejected a NaN/Inf policy action")
        self._raw_actions.copy_(actions)
        target = actions * self._scale + self._offset
        self._processed_actions = torch.maximum(
            self._lower, torch.minimum(self._upper, target)
        )

    def apply_actions(self) -> None:
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = torch.maximum(
            self._lower,
            torch.minimum(self._upper, self._processed_actions - encoder_bias),
        )
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class SafeT1JointPositionActionCfg(BaseActionCfg):
    """Configuration for the canonical fail-closed T1 action term."""

    scale: float = 0.6
    use_default_offset: bool = True
    preserve_order: bool = True

    def build(self, env) -> SafeT1JointPositionAction:
        if not self.use_default_offset:
            raise ValueError("v046 requires the audited default-pose offset")
        return SafeT1JointPositionAction(self, env)
