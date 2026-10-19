"""Persistent per-physics-substep forensics and fail-stop guard for v045."""

from __future__ import annotations

import hashlib
import json
import math
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


WATCHDOG_SCHEMA_VERSION = "v045_physics_substep_watchdog_v2"
REPORT_SCHEMA_VERSION = "v045_stability_report_v2"
METRIC_NAMES = (
    "root_linear_speed_m_s",
    "root_angular_speed_rad_s",
    "joint_velocity_peak_rad_s",
    "joint_acceleration_peak_rad_s2",
    "joint_limit_min_margin_rad",
    "joint_limit_max_violation_rad",
    "actuator_generalized_force_peak_nm",
    "control_peak",
    "contact_constraint_force_peak_n",
    "nefc",
    "njmax",
    "raw_actor_action_peak",
    "processed_delta_peak_rad",
    "final_target_peak_rad",
    "expanded_body_ipos_peak_m",
)


@dataclass(frozen=True)
class StabilityWatchdogCfg:
    """Thresholds and bounded evidence storage for one training process."""

    ring_size_substeps: int = 128
    velocity_warn_rad_s: float = 50.0
    velocity_abort_rad_s: float = 100.0
    velocity_rearm_rad_s: float = 40.0
    # The exact RCSS MJB uses compliant joint constraints.  An 80-episode
    # no-policy hold baseline exceeds 0.0001 rad in every fallen episode and
    # reaches 0.28014 rad.  The watchdog therefore treats 0.30/0.40 rad as the
    # calibrated warning/abort envelope while command targets remain clamped
    # strictly inside the physical ranges by safe_actions.py.
    joint_limit_warn_rad: float = 0.30
    joint_limit_abort_rad: float = 0.40
    joint_limit_rearm_rad: float = 0.25
    joint_limit_count_tolerance_rad: float = 0.30
    constraint_capacity_warn_fraction: float = 0.90
    constraint_capacity_rearm_fraction: float = 0.80
    max_full_snapshots_per_event: int = 5
    max_compact_rows_per_event: int = 32

    def __post_init__(self) -> None:
        if self.ring_size_substeps < 100:
            raise ValueError("v045 watchdog requires at least 100 pre-trigger substeps")
        if not 0.0 < self.velocity_warn_rad_s < self.velocity_abort_rad_s:
            raise ValueError("watchdog velocity warn/abort thresholds are invalid")
        if not 0.0 <= self.joint_limit_warn_rad < self.joint_limit_abort_rad:
            raise ValueError("watchdog joint-limit warn/abort thresholds are invalid")
        if not 0.0 <= self.joint_limit_count_tolerance_rad <= self.joint_limit_warn_rad:
            raise ValueError("watchdog joint-limit count tolerance is invalid")
        if self.max_full_snapshots_per_event <= 0:
            raise ValueError("max_full_snapshots_per_event must be positive")
        if self.max_compact_rows_per_event < self.max_full_snapshots_per_event:
            raise ValueError(
                "max_compact_rows_per_event must cover every full snapshot"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert tensors and non-finite floats into strict JSON evidence."""

    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class V045PhysicsSubstepWatchdog:
    """Observe the state immediately after every real MJWarp physics substep."""

    def __init__(
        self,
        env: Any,
        run_dir: Path,
        *,
        training_profile: dict[str, Any],
        config: StabilityWatchdogCfg | None = None,
    ) -> None:
        self.env = env
        self.run_dir = Path(run_dir).resolve()
        self.output_dir = self.run_dir / "stability"
        self.events_dir = self.output_dir / "events"
        self.cfg = config or StabilityWatchdogCfg()
        self.training_profile = training_profile
        self.robot = env.scene["robot"]
        self.action_term = env.action_manager.get_term("joint_pos")
        for name in ("processed_delta", "final_target", "raw_action"):
            if not hasattr(self.action_term, name):
                raise TypeError(f"v045 action term is missing watchdog telemetry: {name}")
        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.njmax = int(env.cfg.sim.njmax)
        if self.njmax <= 0:
            raise ValueError("v045 watchdog requires a finite positive sim.njmax")

        metric_count = len(METRIC_NAMES)
        self._ring = torch.full(
            (self.cfg.ring_size_substeps, self.num_envs, metric_count),
            float("nan"),
            device=self.device,
            dtype=torch.float32,
        )
        self._ring_steps = [-1] * self.cfg.ring_size_substeps
        self._ring_count = 0
        self._ring_next = 0
        self._physics_substeps_checked = 0
        self._event_file_count = 0
        self._peak = torch.full(
            (metric_count,), -float("inf"), device=self.device, dtype=torch.float32
        )
        self._minimum_joint_margin = torch.tensor(
            float("inf"), device=self.device, dtype=torch.float32
        )
        self._warning_substeps = torch.zeros((), device=self.device, dtype=torch.int64)
        self._abort_substeps = torch.zeros((), device=self.device, dtype=torch.int64)
        self._nonfinite_substeps = torch.zeros((), device=self.device, dtype=torch.int64)
        self._joint_violation_substeps = torch.zeros(
            (), device=self.device, dtype=torch.int64
        )
        self._capacity_near_miss_substeps = torch.zeros(
            (), device=self.device, dtype=torch.int64
        )
        self._velocity_warn_latched = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._joint_warn_latched = torch.zeros_like(self._velocity_warn_latched)
        self._capacity_warn_latched = torch.zeros_like(self._velocity_warn_latched)
        self._episode_index = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        self._previous_episode_length = env.episode_length_buf.clone()
        self._finalized = False
        self._installed = False
        self._original_scene_update = None
        self._original_process_action = None

        source_path = Path(__file__).resolve()
        config_payload = {
            "schema_version": WATCHDOG_SCHEMA_VERSION,
            "hook": (
                "ManagerBasedRlEnv.step -> sim.step -> scene.update -> "
                "V045PhysicsSubstepWatchdog.check_substep"
            ),
            "checks_after_every_physics_substep": True,
            "num_environments": self.num_envs,
            "physics_dt_s": float(env.physics_dt),
            "policy_dt_s": float(env.step_dt),
            "decimation": int(env.cfg.decimation),
            "njmax": self.njmax,
            "metric_names": list(METRIC_NAMES),
            "thresholds_and_storage": asdict(self.cfg),
            "gpu_ring_estimated_bytes": int(self._ring.numel() * self._ring.element_size()),
            "storage_design": (
                "bounded compact all-environment metric ring on training device; "
                "full state/action/body snapshots copied to CPU only on an event"
            ),
            "training_profile": training_profile,
            "watchdog_source": {"path": str(source_path), "sha256": _sha256(source_path)},
        }
        _write_json_exclusive(self.output_dir / "watchdog_config.json", config_payload)

    def install(self) -> V045PhysicsSubstepWatchdog:
        """Install the true substep hook and a pre-physics action guard."""

        if self._installed:
            raise RuntimeError("v045 watchdog is already installed")
        self._original_scene_update = self.env.scene.update
        self._original_process_action = self.env.action_manager.process_action

        def guarded_scene_update(*args: Any, **kwargs: Any) -> Any:
            assert self._original_scene_update is not None
            result = self._original_scene_update(*args, **kwargs)
            self.check_substep()
            return result

        def guarded_process_action(action: torch.Tensor) -> Any:
            assert self._original_process_action is not None
            try:
                result = self._original_process_action(action)
            except BaseException as exc:
                self.record_prephysics_failure(
                    "action_processing_exception", action=action, exception=exc
                )
                raise
            action_finite = torch.isfinite(self.action_term.raw_action).all(dim=1)
            delta_finite = torch.isfinite(self.action_term.processed_delta).all(dim=1)
            target_finite = torch.isfinite(self.action_term.final_target).all(dim=1)
            invalid = ~(action_finite & delta_finite & target_finite)
            if bool(invalid.any()):
                self._persist_event(
                    kinds=("nonfinite_action_or_target",),
                    event_mask=invalid,
                    metrics=None,
                    prephysics_action=action,
                )
                self._nonfinite_substeps += invalid.sum()
                raise FloatingPointError(
                    "v045 watchdog found a non-finite action/delta/final target"
                )
            return result

        self.env.scene.update = guarded_scene_update
        self.env.action_manager.process_action = guarded_process_action
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        assert self._original_scene_update is not None
        assert self._original_process_action is not None
        self.env.scene.update = self._original_scene_update
        self.env.action_manager.process_action = self._original_process_action
        self._installed = False

    def _update_episode_indexes(self) -> None:
        current = self.env.episode_length_buf
        reset = current < self._previous_episode_length
        self._episode_index += reset.to(torch.int64)
        self._previous_episode_length.copy_(current)

    def _contact_peak(self) -> torch.Tensor:
        forces = self.env.sim.data.efc.force
        types = self.env.sim.data.efc.type
        nefc = self.env.sim.data.nefc.to(torch.int64)
        row = torch.arange(self.njmax, device=self.device).unsqueeze(0)
        valid = row < nefc.unsqueeze(1)
        # MuJoCo constraint types 5, 6 and 7 are frictionless, pyramidal and
        # elliptic contacts respectively.  Lower values are equality/friction/
        # joint-limit constraints and must not be mislabeled as contact force.
        contact = valid & (types >= 5)
        return torch.where(contact, forces.abs(), 0.0).amax(dim=1)

    def _expanded_body_ipos(self) -> torch.Tensor:
        """Return model body positions with one row per simulated environment."""

        values = self.env.sim.model.body_ipos
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.shape[0] == 1:
            return values.expand(self.num_envs, -1, -1)
        if values.shape[0] != self.num_envs:
            raise RuntimeError(
                "v045 watchdog cannot map body_ipos to the environment batch: "
                f"shape={tuple(values.shape)}, num_envs={self.num_envs}"
            )
        return values

    def _metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        qvel = self.env.sim.data.qvel
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        joint_acc = self.robot.data.joint_acc
        limits = torch.stack(
            (
                self.action_term.deployment_lower_limits.expand(self.num_envs, -1),
                self.action_term.deployment_upper_limits.expand(self.num_envs, -1),
            ),
            dim=-1,
        )
        margin = torch.minimum(joint_pos - limits[..., 0], limits[..., 1] - joint_pos)
        violation = torch.relu(-margin)
        raw_action = self.action_term.raw_action
        delta = self.action_term.processed_delta
        final_target = self.action_term.final_target
        body_ipos = self._expanded_body_ipos()
        nefc = self.env.sim.data.nefc.to(torch.float32)
        metrics = torch.stack(
            (
                torch.linalg.vector_norm(qvel[:, 0:3], dim=1),
                torch.linalg.vector_norm(qvel[:, 3:6], dim=1),
                joint_vel.abs().amax(dim=1),
                joint_acc.abs().amax(dim=1),
                margin.amin(dim=1),
                violation.amax(dim=1),
                self.robot.data.qfrc_actuator.abs().amax(dim=1),
                self.env.sim.data.ctrl.abs().amax(dim=1),
                self._contact_peak(),
                nefc,
                torch.full_like(nefc, float(self.njmax)),
                raw_action.abs().amax(dim=1),
                delta.abs().amax(dim=1),
                final_target.abs().amax(dim=1),
                body_ipos.abs().flatten(1).amax(dim=1),
            ),
            dim=1,
        ).to(torch.float32)
        finite_tensors = (
            self.env.sim.data.qpos,
            qvel,
            self.env.sim.data.qacc,
            self.env.sim.data.qfrc_actuator,
            self.env.sim.data.ctrl,
            joint_pos,
            joint_vel,
            joint_acc,
            limits,
            raw_action,
            delta,
            final_target,
            body_ipos,
            self.env.sim.data.efc.force,
        )
        finite = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        for tensor in finite_tensors:
            finite &= torch.isfinite(tensor).flatten(1).all(dim=1)
        return metrics, finite

    def _ring_for_env(self, env_id: int) -> list[dict[str, Any]]:
        if self._ring_count == 0:
            return []
        count = min(self._ring_count, self.cfg.ring_size_substeps)
        start = self._ring_next if self._ring_count >= self.cfg.ring_size_substeps else 0
        indexes = [
            (start + offset) % self.cfg.ring_size_substeps for offset in range(count)
        ]
        values = self._ring[indexes, env_id].detach().cpu().tolist()
        return [
            {
                "physics_step": int(self._ring_steps[index]),
                "metrics": dict(zip(METRIC_NAMES, row, strict=True)),
            }
            for index, row in zip(indexes, values, strict=True)
        ]

    def _snapshot(self, env_id: int) -> dict[str, Any]:
        nefc = int(self.env.sim.data.nefc[env_id].item())
        pose_values = self.env.extras.get("v045_reset_pose_index")
        pose_index = int(pose_values[env_id].item()) if pose_values is not None else -1
        return {
            "environment_id": env_id,
            "episode_index": int(self._episode_index[env_id].item()),
            "episode_control_step": int(self.env.episode_length_buf[env_id].item()),
            "global_control_step": int(self.env.common_step_counter),
            "global_physics_step": int(self.env._sim_step_counter),
            "reset_pose_index": pose_index,
            "raw_actor_action": self.action_term.raw_action[env_id],
            "processed_delta": self.action_term.processed_delta[env_id],
            "final_target": self.action_term.final_target[env_id],
            "expanded_body_ipos": self._expanded_body_ipos()[env_id],
            "qpos": self.env.sim.data.qpos[env_id],
            "qvel": self.env.sim.data.qvel[env_id],
            "qacc": self.env.sim.data.qacc[env_id],
            "joint_pos": self.robot.data.joint_pos[env_id],
            "joint_vel": self.robot.data.joint_vel[env_id],
            "joint_acc": self.robot.data.joint_acc[env_id],
            "joint_pos_limits": self.robot.data.joint_pos_limits[env_id],
            "deployment_joint_pos_limits": torch.stack(
                (
                    self.action_term.deployment_lower_limits[0],
                    self.action_term.deployment_upper_limits[0],
                ),
                dim=-1,
            ),
            "qfrc_actuator": self.env.sim.data.qfrc_actuator[env_id],
            "ctrl": self.env.sim.data.ctrl[env_id],
            "nefc": nefc,
            "njmax": self.njmax,
            "efc_force_valid": self.env.sim.data.efc.force[env_id, :nefc],
            "efc_type_valid": self.env.sim.data.efc.type[env_id, :nefc],
            "pretrigger_ring": self._ring_for_env(env_id),
        }

    def _persist_event(
        self,
        *,
        kinds: tuple[str, ...],
        event_mask: torch.Tensor,
        metrics: torch.Tensor | None,
        prephysics_action: torch.Tensor | None = None,
    ) -> None:
        env_ids = event_mask.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
        if not env_ids:
            return
        self._event_file_count += 1
        snapshot_ids = env_ids[: self.cfg.max_full_snapshots_per_event]
        compact_ids = env_ids[: self.cfg.max_compact_rows_per_event]
        current_metrics = None
        if metrics is not None:
            rows = metrics[compact_ids].detach().cpu().tolist()
            current_metrics = [
                {
                    "environment_id": int(env_id),
                    "metrics": dict(zip(METRIC_NAMES, row, strict=True)),
                }
                for env_id, row in zip(compact_ids, rows, strict=True)
            ]
        compact_prephysics_action = None
        if prephysics_action is not None:
            compact_prephysics_action = {
                "environment_ids": compact_ids,
                "values": prephysics_action[compact_ids],
            }
        payload = {
            "schema_version": WATCHDOG_SCHEMA_VERSION,
            "event_index": self._event_file_count,
            "kinds": list(kinds),
            "global_physics_step": int(self.env._sim_step_counter),
            "global_control_step": int(self.env.common_step_counter),
            "triggered_environment_count": len(env_ids),
            "triggered_environment_ids": env_ids,
            "compact_row_environment_ids": compact_ids,
            "compact_row_cap": self.cfg.max_compact_rows_per_event,
            "full_snapshot_environment_ids": snapshot_ids,
            "full_snapshot_cap": self.cfg.max_full_snapshots_per_event,
            "current_metrics": current_metrics,
            "prephysics_action_argument": compact_prephysics_action,
            "full_snapshots": [self._snapshot(int(env_id)) for env_id in snapshot_ids],
            "available_pretrigger_substeps": min(
                self._ring_count, self.cfg.ring_size_substeps
            ),
        }
        name = f"event_{self._event_file_count:06d}_{kinds[0]}.json"
        _write_json_exclusive(self.events_dir / name, payload)

    def check_substep(self) -> None:
        """Check one post-``scene.update`` 5 ms state and fail immediately if needed."""

        self._physics_substeps_checked += 1
        self._update_episode_indexes()
        metrics, finite = self._metrics()
        finite_metrics = torch.nan_to_num(
            metrics, nan=-float("inf"), posinf=float("inf"), neginf=float("inf")
        )
        self._peak = torch.maximum(self._peak, finite_metrics.amax(dim=0))
        finite_margin = torch.nan_to_num(
            metrics[:, 4], nan=float("inf"), posinf=float("inf"), neginf=-float("inf")
        )
        self._minimum_joint_margin = torch.minimum(
            self._minimum_joint_margin, finite_margin.amin()
        )

        speed = torch.maximum(metrics[:, 1], metrics[:, 2])
        velocity_warn = speed >= self.cfg.velocity_warn_rad_s
        velocity_abort = speed >= self.cfg.velocity_abort_rad_s
        joint_violation = metrics[:, 5]
        joint_warn = joint_violation > self.cfg.joint_limit_warn_rad
        joint_abort = joint_violation > self.cfg.joint_limit_abort_rad
        capacity_fraction = metrics[:, 9] / float(self.njmax)
        capacity_warn = capacity_fraction >= self.cfg.constraint_capacity_warn_fraction
        capacity_abort = metrics[:, 9] >= float(self.njmax)
        nonfinite = ~finite
        warning = velocity_warn | joint_warn | capacity_warn
        abort = velocity_abort | joint_abort | capacity_abort | nonfinite
        self._warning_substeps += warning.sum()
        self._abort_substeps += abort.sum()
        self._nonfinite_substeps += nonfinite.sum()
        self._joint_violation_substeps += (
            joint_violation > self.cfg.joint_limit_count_tolerance_rad
        ).sum()
        self._capacity_near_miss_substeps += capacity_warn.sum()

        velocity_edge = velocity_warn & ~self._velocity_warn_latched
        joint_edge = joint_warn & ~self._joint_warn_latched
        capacity_edge = capacity_warn & ~self._capacity_warn_latched
        event_mask = velocity_edge | joint_edge | capacity_edge | abort
        abort_any = False
        if bool(event_mask.any()):
            abort_any = bool(abort.any())
            kinds = []
            if bool(nonfinite.any()):
                kinds.append("nonfinite_state")
            if bool(velocity_abort.any()):
                kinds.append("velocity_abort")
            elif bool(velocity_edge.any()):
                kinds.append("velocity_near_miss")
            if bool(joint_abort.any()):
                kinds.append("joint_limit_abort")
            elif bool(joint_edge.any()):
                kinds.append("joint_limit_near_miss")
            if bool(capacity_abort.any()):
                kinds.append("constraint_capacity_exhausted")
            elif bool(capacity_edge.any()):
                kinds.append("constraint_capacity_near_miss")
            self._persist_event(kinds=tuple(kinds), event_mask=event_mask, metrics=metrics)

        self._ring[self._ring_next].copy_(metrics)
        self._ring_steps[self._ring_next] = int(self.env._sim_step_counter)
        self._ring_next = (self._ring_next + 1) % self.cfg.ring_size_substeps
        self._ring_count += 1
        self._velocity_warn_latched = torch.where(
            speed < self.cfg.velocity_rearm_rad_s,
            torch.zeros_like(self._velocity_warn_latched),
            self._velocity_warn_latched | velocity_warn,
        )
        self._joint_warn_latched = torch.where(
            joint_violation < self.cfg.joint_limit_rearm_rad,
            torch.zeros_like(self._joint_warn_latched),
            self._joint_warn_latched | joint_warn,
        )
        self._capacity_warn_latched = torch.where(
            capacity_fraction < self.cfg.constraint_capacity_rearm_fraction,
            torch.zeros_like(self._capacity_warn_latched),
            self._capacity_warn_latched | capacity_warn,
        )
        if abort_any:
            raise FloatingPointError(
                "v045 physics-substep watchdog abort: "
                f"{int(abort.sum().item())} environment(s) crossed a hard threshold"
            )

    def record_prephysics_failure(
        self, kind: str, *, action: torch.Tensor, exception: BaseException
    ) -> None:
        finite = torch.isfinite(action).flatten(1).all(dim=1)
        event_mask = ~finite
        if not bool(event_mask.any()):
            event_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._persist_event(
            kinds=(kind, type(exception).__name__),
            event_mask=event_mask,
            metrics=None,
            prephysics_action=action,
        )

    def finalize(
        self, *, status: str, exception: BaseException | None = None
    ) -> Path:
        """Write the one immutable run-level report used by formal gates."""

        report_path = self.output_dir / "stability_report.json"
        if self._finalized:
            return report_path
        self._finalized = True
        self.uninstall()
        peak_values = self._peak.detach().cpu().tolist()
        peaks = dict(zip(METRIC_NAMES, peak_values, strict=True))
        counts = {
            "warning_environment_substeps": int(self._warning_substeps.item()),
            "abort_environment_substeps": int(self._abort_substeps.item()),
            "nonfinite_environment_substeps": int(self._nonfinite_substeps.item()),
            "joint_limit_violation_environment_substeps": int(
                self._joint_violation_substeps.item()
            ),
            "constraint_capacity_near_miss_environment_substeps": int(
                self._capacity_near_miss_substeps.item()
            ),
            "persistent_event_files": self._event_file_count,
        }
        clean = bool(
            status == "completed"
            and counts["warning_environment_substeps"] == 0
            and counts["abort_environment_substeps"] == 0
            and counts["nonfinite_environment_substeps"] == 0
            and counts["joint_limit_violation_environment_substeps"] == 0
            and counts["constraint_capacity_near_miss_environment_substeps"] == 0
            and counts["persistent_event_files"] == 0
        )
        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "watchdog_schema_version": WATCHDOG_SCHEMA_VERSION,
            "status": status,
            "completed_cleanly_with_zero_events": clean,
            "physics_substeps_checked": self._physics_substeps_checked,
            "num_environments": self.num_envs,
            "environment_substeps_checked": self._physics_substeps_checked
            * self.num_envs,
            "ring_size_substeps": self.cfg.ring_size_substeps,
            "per_physics_substep_hook": True,
            "thresholds": asdict(self.cfg),
            "peaks": peaks,
            "minimum_joint_limit_margin_rad": float(
                self._minimum_joint_margin.detach().cpu().item()
            ),
            "counts": counts,
            "training_profile": self.training_profile,
            "watchdog_config": {
                "path": str((self.output_dir / "watchdog_config.json").resolve()),
                "sha256": _sha256(self.output_dir / "watchdog_config.json"),
            },
            "exception": (
                None
                if exception is None
                else {
                    "type": type(exception).__name__,
                    "message": str(exception),
                    "traceback": "".join(
                        traceback.format_exception(
                            type(exception), exception, exception.__traceback__
                        )
                    ),
                }
            ),
            "formal_gate_rule": (
                "status=completed, true substep hook, at least one checked substep, "
                "zero warning/abort/nonfinite/joint-violation/capacity events, and "
                "zero persistent event files"
            ),
        }
        _write_json_exclusive(report_path, payload)
        return report_path


def install_v045_watchdog(
    env: Any,
    run_dir: Path,
    *,
    training_profile: dict[str, Any],
    config: StabilityWatchdogCfg | None = None,
) -> V045PhysicsSubstepWatchdog:
    return V045PhysicsSubstepWatchdog(
        env, run_dir, training_profile=training_profile, config=config
    ).install()
