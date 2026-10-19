"""Deployment controller for v045 policies under the shared actuator contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.v044_mjlab_t1_getup.rcss_getup_controller import (
    RcssMjlabGetupController as _SourceController,
)
from experiments.v044_mjlab_t1_getup.mjlab_getup_policy import MjlabGetupPolicy
from experiments.v045_rcss_t1_getup_adaptation.actuator_contract import (
    RCSS_EFFECTIVE_EFFORT_LIMIT,
    V045_GETUP_KD,
    V045_GETUP_KP,
)
from locomotion.rcss_t1_controller import read_joint_state
from locomotion.t1_spec import T1_ACTION_DELTA_LIMIT_RAD, T1_MOTOR_NAMES


POSE_NAMES = ("prone", "supine", "left_side", "right_side")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ensemble_manifest(
    path: Path,
) -> tuple[dict[str, Path], dict[str, float], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v045_pose_routed_ensemble_v1":
        raise ValueError("unsupported v045 get-up ensemble manifest")
    policies = payload.get("policies")
    fallback_pose = payload.get("fallback_pose")
    if not isinstance(policies, dict) or set(policies) != set(POSE_NAMES):
        raise ValueError("ensemble manifest must define exactly four pose policies")
    if fallback_pose not in POSE_NAMES:
        raise ValueError("ensemble fallback pose is invalid")
    resolved: dict[str, Path] = {}
    action_scales: dict[str, float] = {}
    for pose in POSE_NAMES:
        reference = policies[pose]
        if not isinstance(reference, dict):
            raise ValueError(f"ensemble policy reference is invalid: {pose}")
        checkpoint = Path(str(reference.get("path", ""))).expanduser().resolve()
        if not checkpoint.is_file() or _sha256(checkpoint) != reference.get("sha256"):
            raise ValueError(f"ensemble policy file/hash changed: {pose}")
        action_scale = float(reference.get("action_scale_rad", 0.6))
        if not np.isfinite(action_scale) or not 0.0 < action_scale <= 0.6:
            raise ValueError(f"ensemble action scale is invalid: {pose}")
        resolved[pose] = checkpoint
        action_scales[pose] = action_scale
    return resolved, action_scales, str(fallback_pose)


class RcssMjlabGetupController(_SourceController):
    """Run the v045 actor through RCSS's native position-PD actuator path."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        effort_limits: np.ndarray = RCSS_EFFECTIVE_EFFORT_LIMIT,
        max_target_delta_rad: float = T1_ACTION_DELTA_LIMIT_RAD,
    ) -> None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        ensemble_paths: dict[str, Path] | None = None
        ensemble_action_scales: dict[str, float] | None = None
        fallback_pose: str | None = None
        if checkpoint_path.suffix.lower() == ".json":
            ensemble_paths, ensemble_action_scales, fallback_pose = (
                _load_ensemble_manifest(checkpoint_path)
            )
            source_checkpoint = ensemble_paths[fallback_pose]
        else:
            source_checkpoint = checkpoint_path
        super().__init__(
            source_checkpoint,
            device=device,
            effort_limits=effort_limits,
            max_target_delta_rad=max_target_delta_rad,
        )
        self.ensemble_manifest = checkpoint_path if ensemble_paths is not None else None
        self._fallback_pose = fallback_pose
        self._action_scales_by_pose = ensemble_action_scales
        self._policies_by_pose: dict[str, MjlabGetupPolicy] | None = None
        self.selected_pose: str | None = None
        if ensemble_paths is not None:
            self._policies_by_pose = {}
            for pose, policy_checkpoint in ensemble_paths.items():
                self._policies_by_pose[pose] = (
                    self.policy
                    if pose == fallback_pose
                    else MjlabGetupPolicy(policy_checkpoint, device=device)
                )

    def reset(self, joint_pos: np.ndarray) -> None:
        if self._policies_by_pose is None:
            super().reset(joint_pos)
            return
        if self._fallback_pose is None:
            raise RuntimeError("ensemble controller has no fallback policy")
        self.policy = self._policies_by_pose[self._fallback_pose]
        if self._action_scales_by_pose is None:
            raise RuntimeError("ensemble controller has no action-scale mapping")
        self.action_scale = self._action_scales_by_pose[self._fallback_pose]
        super().reset(joint_pos)
        for pose, policy in self._policies_by_pose.items():
            if pose != self._fallback_pose:
                policy.reset()
        self.selected_pose = None

    def policy_step(self, simulation: object, agent: object) -> np.ndarray:
        if self._policies_by_pose is not None and self.selected_pose is None:
            rotation = np.asarray(agent.xmat, dtype=np.float32).reshape(3, 3)
            projected_gravity = rotation.T @ np.asarray(
                [0.0, 0.0, -1.0], dtype=np.float32
            )
            horizontal = projected_gravity[:2]
            if float(np.linalg.norm(horizontal)) < 0.08:
                if self._fallback_pose is None:
                    raise RuntimeError("ensemble controller has no fallback policy")
                selected = self._fallback_pose
            elif abs(float(horizontal[0])) >= abs(float(horizontal[1])):
                selected = "prone" if horizontal[0] >= 0.0 else "supine"
            else:
                selected = "left_side" if horizontal[1] >= 0.0 else "right_side"
            self.selected_pose = selected
            self.policy = self._policies_by_pose[selected]
            if self._action_scales_by_pose is None:
                raise RuntimeError("ensemble controller has no action-scale mapping")
            self.action_scale = self._action_scales_by_pose[selected]
        return super().policy_step(simulation, agent)

    def physics_step(self, simulation: object, agent: object) -> dict[str, float]:
        joint_pos, joint_vel = read_joint_state(agent)
        requested_torque = (
            V045_GETUP_KP * (self.target - joint_pos) - V045_GETUP_KD * joint_vel
        )
        if not np.isfinite(requested_torque).all():
            raise FloatingPointError("non-finite v045 get-up torque")
        # Keep an explicit capped estimate for the safety recorder.  The actual
        # command must use RCSS's native position/Kp/Kd interface: training uses
        # a BuiltinPositionActuator, and the manual external-torque interface is
        # not dynamically equivalent in RCSSServerMJ.
        sent_torque = np.clip(
            requested_torque, -self.effort_limits, self.effort_limits
        ).astype(np.float32)
        prefix = agent.agent_id.prefix
        for motor, target, kp, kd in zip(
            T1_MOTOR_NAMES,
            self.target,
            V045_GETUP_KP,
            V045_GETUP_KD,
            strict=True,
        ):
            simulation.ctrl_motor(
                prefix + motor,
                float(target),
                0.0,
                float(kp),
                float(kd),
                0.0,
            )
        self.last_requested_torque[:] = requested_torque
        self.last_sent_torque[:] = sent_torque
        return {
            "saturation_fraction": float(
                np.mean(np.abs(requested_torque) >= self.effort_limits * 0.999)
            ),
            "maximum_requested_torque_nm": float(np.max(np.abs(requested_torque))),
            "maximum_sent_torque_nm": float(np.max(np.abs(sent_torque))),
        }
