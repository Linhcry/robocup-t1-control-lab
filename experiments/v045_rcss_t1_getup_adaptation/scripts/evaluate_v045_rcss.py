"""Fail-closed RCSS evaluation for exact-target v045 getup checkpoints.

Formal promotion always evaluates all four fallen poses, requires an independent
fixed-validation proof before opening the held-out split, sustains standing for
three seconds, hands off to the existing locomotion actor for five seconds and
requires one metre of *forward* displacement.  Every physics step contributes
joint, contact and actuator-torque evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_ACCEPTANCE = (
    EXPERIMENT_ROOT / "configs" / "v045_competitive_acceptance_v3.json"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v044_mjlab_t1_getup.mjlab_getup_policy import (  # noqa: E402
    HOME_QPOS,
    sha256_file,
)
from experiments.v045_rcss_t1_getup_adaptation.rcss_getup_controller import (  # noqa: E402
    RCSS_EFFECTIVE_EFFORT_LIMIT,
    RcssMjlabGetupController,
)
from locomotion.official_t1_policy import MotionCommand  # noqa: E402
from locomotion.rcss_simulation import make_single_t1_simulation, torso_upright  # noqa: E402
from locomotion.rcss_t1_controller import (  # noqa: E402
    COMMON_KD,
    COMMON_KP,
    PARALLEL_MECH_INDEXES,
    PREPARE_KD,
    PREPARE_KP,
    PREPARE_QPOS,
    TORQUE_LIMIT,
    RcssT1Controller,
    read_joint_state,
)
from locomotion.t1_spec import (  # noqa: E402
    T1_ACTION_DELTA_LIMIT_RAD,
    T1_JOINT_NAMES,
    T1_LOWER_LIMITS,
    T1_SAFE_HARD_LOWER_LIMITS,
    T1_SAFE_HARD_UPPER_LIMITS,
    T1_UPPER_LIMITS,
    clamp_t1_targets,
    filter_t1_dynamic_targets,
)


DEFAULT_LOCOMOTION_MODEL = Path(
    "/home/linh/robot_soccer/third_party/booster_gym/deploy/models/T1.pt"
)
POSES = {
    "prone": ((0.0, 1.0, 0.0), math.pi / 2.0),
    "supine": ((0.0, 1.0, 0.0), -math.pi / 2.0),
    "left_side": ((1.0, 0.0, 0.0), -math.pi / 2.0),
    "right_side": ((1.0, 0.0, 0.0), math.pi / 2.0),
}
POLICY_DT = 0.02
WALK_COMMAND_VX_MPS = 0.30
JOINT_SPACE_SATURATION_MASK = np.ones(23, dtype=bool)
JOINT_SPACE_SATURATION_MASK[list(PARALLEL_MECH_INDEXES)] = False
CONTACT_REGIONS = (
    "head",
    "torso_or_waist",
    "arms",
    "hands",
    "upper_legs",
    "lower_legs",
    "feet",
    "other",
)


def _sha256_json(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_fingerprint(seeds: list[int]) -> str:
    encoded = json.dumps(seeds, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_acceptance(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "required_splits",
        "required_initial_poses",
        "success_definition",
        "safety_thresholds",
        "quality_gates",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"acceptance config is missing keys: {sorted(missing)}")
    if config["required_initial_poses"] != list(POSES):
        raise ValueError("acceptance pose order does not match evaluator pose order")
    for split in ("fixed_validation", "held_out_test"):
        split_cfg = config["required_splits"].get(split, {})
        if "seed_start" not in split_cfg or "episodes_per_pose" not in split_cfg:
            raise ValueError(f"acceptance split is incomplete: {split}")
    if config.get("schema_version") != "v045_getup_acceptance_v3":
        raise ValueError("v045 evaluator requires the calibrated v3 acceptance schema")
    calibration = config.get("physical_constraint_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("acceptance config lacks physical-constraint calibration")
    calibration_path = (path.parent / calibration.get("path", "")).resolve()
    if (
        not calibration_path.is_file()
        or _sha256_json(calibration_path) != calibration.get("sha256")
    ):
        raise ValueError("physical-constraint calibration file/hash changed")
    calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        calibration_payload.get("schema_version")
        != "v045_passive_hold_limit_probe_v1"
        or calibration_payload.get("aggregate", {}).get("episode_count") != 80
        or calibration_payload.get("aggregate", {}).get("violation_rate") != 1.0
    ):
        raise ValueError("physical-constraint calibration contract is invalid")
    threshold = float(
        config["safety_thresholds"][
            "physical_joint_constraint_excursion_tolerance_rad"
        ]
    )
    sampled_max = float(
        calibration_payload["aggregate"]["maximum_joint_limit_violation_rad"]
    )
    if not sampled_max < threshold <= 0.30:
        raise ValueError("physical excursion envelope is not calibration-bound")
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("fixed_validation", "held_out_test"),
        default="fixed_validation",
    )
    parser.add_argument("--fixed-proof", type=Path)
    parser.add_argument("--candidate-proof", type=Path)
    parser.add_argument("--acceptance-config", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--locomotion-model", type=Path, default=DEFAULT_LOCOMOTION_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--poses", nargs="+", choices=tuple(POSES))
    parser.add_argument("--episodes-per-pose", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--settle-s", type=float)
    parser.add_argument("--recovery-timeout-s", type=float)
    parser.add_argument(
        "--engineering-max-target-delta-rad",
        type=float,
        help=(
            "non-promotable smoke-only override used to scan the target-rate "
            "safety/skill trade-off"
        ),
    )
    parser.add_argument(
        "--engineering-tilt-deg",
        type=int,
        choices=(0, 5, 10, 12, 14, 15, 30, 45, 60, 75, 90),
        help=(
            "smoke-only initial tilt used to verify a reverse-curriculum stage; "
            "0 degrees tests perturbed upright balance"
        ),
    )
    parser.add_argument(
        "--engineering-action-source",
        choices=(
            "checkpoint",
            "zero_delta",
            "home_target",
            "prepare_target",
            "prepare_pd",
        ),
        default="checkpoint",
        help=(
            "smoke-only control probe; zero_delta holds the measured pose and "
            "home_target rate-limits toward the RCSS home posture, prepare_target "
            "uses the preparation posture through the manual-torque path, and "
            "prepare_pd tests the existing locomotion actuator interface"
        ),
    )
    args = parser.parse_args()
    if args.episodes_per_pose is not None and args.episodes_per_pose <= 0:
        parser.error("episodes-per-pose must be positive")
    if args.smoke and args.split != "fixed_validation":
        parser.error("engineering smoke is allowed only against the fixed split")
    if not args.smoke and args.split == "held_out_test" and args.fixed_proof is None:
        parser.error("formal held_out_test requires --fixed-proof")
    if not args.smoke and args.candidate_proof is None:
        parser.error("formal RCSS evaluation requires --candidate-proof")
    if args.engineering_max_target_delta_rad is not None:
        if not args.smoke:
            parser.error("target-rate override is allowed only with --smoke")
        if (
            not math.isfinite(args.engineering_max_target_delta_rad)
            or args.engineering_max_target_delta_rad <= 0.0
        ):
            parser.error("engineering target-rate override must be positive and finite")
    if args.engineering_tilt_deg is not None and not args.smoke:
        parser.error("initial-tilt override is allowed only with --smoke")
    if args.engineering_action_source != "checkpoint" and not args.smoke:
        parser.error("engineering action-source probes require --smoke")
    return args


def _resolve_protocol(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    split_cfg = config["required_splits"][args.split]
    success_cfg = config["success_definition"]
    required_poses = list(config["required_initial_poses"])
    poses = list(args.poses) if args.poses is not None else required_poses
    episodes_per_pose = (
        int(args.episodes_per_pose)
        if args.episodes_per_pose is not None
        else int(split_cfg["episodes_per_pose"])
    )
    seed_start = (
        int(args.seed_start)
        if args.seed_start is not None
        else int(split_cfg["seed_start"])
    )
    settle_s = (
        float(args.settle_s)
        if args.settle_s is not None
        else float(success_cfg["settle_s"])
    )
    recovery_timeout_s = (
        float(args.recovery_timeout_s)
        if args.recovery_timeout_s is not None
        else float(success_cfg["recovery_timeout_s"])
    )
    max_target_delta_rad = (
        float(args.engineering_max_target_delta_rad)
        if args.engineering_max_target_delta_rad is not None
        else T1_ACTION_DELTA_LIMIT_RAD
    )
    tilt_deg = (
        int(args.engineering_tilt_deg)
        if args.engineering_tilt_deg is not None
        else 90
    )
    if settle_s < 0.0 or recovery_timeout_s <= 0.0:
        raise ValueError("settle and recovery durations are invalid")
    exact = bool(
        poses == required_poses
        and episodes_per_pose == int(split_cfg["episodes_per_pose"])
        and seed_start == int(split_cfg["seed_start"])
        and math.isclose(settle_s, float(success_cfg["settle_s"]))
        and math.isclose(
            recovery_timeout_s, float(success_cfg["recovery_timeout_s"])
        )
        and math.isclose(max_target_delta_rad, T1_ACTION_DELTA_LIMIT_RAD)
        and tilt_deg == 90
        and args.engineering_action_source == "checkpoint"
    )
    if not args.smoke and not exact:
        raise ValueError(
            "formal evaluation cannot override poses, count, seeds or durations; "
            "use --smoke for a non-promotable engineering run"
        )
    return {
        "formal": bool(not args.smoke and exact),
        "poses": poses,
        "episodes_per_pose": episodes_per_pose,
        "seed_start": seed_start,
        "settle_s": settle_s,
        "recovery_timeout_s": recovery_timeout_s,
        "max_target_delta_rad": max_target_delta_rad,
        "tilt_deg": tilt_deg,
        "action_source": args.engineering_action_source,
    }


def _next_output(checkpoint: Path, label: str) -> Path:
    for run in range(1, 1000):
        candidate = (
            EXPERIMENT_ROOT
            / "metrics"
            / f"v045_{checkpoint.stem}_{label}_run_{run:02d}.json"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError("no free v045 metric run number")


def _validate_fixed_proof(
    path: Path,
    *,
    checkpoint_sha256: str,
    locomotion_sha256: str,
    acceptance_sha256: str,
    candidate_proof_sha256: str,
    held_out_seeds: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != "v045_rcss_getup_eval_v3":
        raise ValueError("fixed proof schema mismatch")
    if proof.get("evaluation_split") != "fixed_validation":
        raise ValueError("fixed proof has the wrong split")
    if not proof.get("formal_protocol", False):
        raise ValueError("fixed proof was not a formal protocol run")
    if not proof.get("validation_passed", False):
        raise ValueError("fixed proof did not pass")
    proof_gates = proof.get("gates")
    if not isinstance(proof_gates, dict) or not proof_gates or not all(
        bool(value) for value in proof_gates.values()
    ):
        raise ValueError("fixed proof gates are incomplete or contain a failure")
    if proof.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("fixed proof checkpoint hash mismatch")
    if proof.get("locomotion_model", {}).get("sha256") != locomotion_sha256:
        raise ValueError("fixed proof locomotion model hash mismatch")
    if proof.get("acceptance_config", {}).get("sha256") != acceptance_sha256:
        raise ValueError("fixed proof acceptance config hash mismatch")
    if (
        proof.get("candidate_selection_proof", {}).get("sha256")
        != candidate_proof_sha256
    ):
        raise ValueError("fixed proof candidate-selection hash mismatch")
    fixed_cfg = config["required_splits"]["fixed_validation"]
    if proof.get("seed_start") != int(fixed_cfg["seed_start"]):
        raise ValueError("fixed proof seed start mismatch")
    if proof.get("episodes_per_pose") != int(fixed_cfg["episodes_per_pose"]):
        raise ValueError("fixed proof episode count mismatch")
    if proof.get("poses") != list(config["required_initial_poses"]):
        raise ValueError("fixed proof pose set mismatch")
    fixed_seeds = [int(e["seed"]) for e in proof.get("episodes", [])]
    expected_count = len(config["required_initial_poses"]) * int(
        fixed_cfg["episodes_per_pose"]
    )
    if len(fixed_seeds) != expected_count or len(set(fixed_seeds)) != expected_count:
        raise ValueError("fixed proof seed evidence is incomplete or duplicated")
    if proof.get("episode_seed_sha256") != _seed_fingerprint(fixed_seeds):
        raise ValueError("fixed proof seed fingerprint mismatch")
    expected_fixed_seeds = [
        int(fixed_cfg["seed_start"])
        + pose_index * int(fixed_cfg["episodes_per_pose"])
        + episode_index
        for pose_index, _ in enumerate(config["required_initial_poses"])
        for episode_index in range(int(fixed_cfg["episodes_per_pose"]))
    ]
    if fixed_seeds != expected_fixed_seeds:
        raise ValueError("fixed proof episode seeds or ordering were altered")
    if not all(
        isinstance(episode.get("safety"), dict)
        and "contact" in episode["safety"]
        and "joint" in episode["safety"]
        and "torque" in episode["safety"]
        and "episode_complete" in episode
        for episode in proof.get("episodes", [])
    ):
        raise ValueError("fixed proof is missing per-episode safety evidence")
    overlap = sorted(set(fixed_seeds).intersection(held_out_seeds))
    if overlap:
        raise ValueError(f"fixed and held-out seeds overlap: {overlap[:5]}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_json(path),
        "fixed_seed_sha256": _seed_fingerprint(fixed_seeds),
        "seed_sets_disjoint": True,
    }


def _validate_candidate_proof(path: Path, checkpoint: Path) -> dict[str, Any]:
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != "v045_target_candidate_eval_v1":
        raise ValueError("candidate proof schema mismatch")
    if not proof.get("formal_target_shortlist_protocol", False):
        raise ValueError("candidate proof was not the formal shortlist")
    if proof.get("seeds") != [45100, 45101, 45102]:
        raise ValueError("candidate proof seed set mismatch")
    if proof.get("num_environments") != 256 or proof.get("steps") != 299:
        raise ValueError("candidate proof evaluation size mismatch")

    selected = proof.get("selected_checkpoint_for_rcss_fixed_validation")
    checkpoint_sha = sha256_file(checkpoint)
    if not isinstance(selected, dict):
        raise ValueError("candidate proof has no selected checkpoint")
    if Path(selected.get("path", "")).resolve() != checkpoint:
        raise ValueError("RCSS checkpoint differs from selected candidate")
    if selected.get("sha256") != checkpoint_sha:
        raise ValueError("selected candidate checkpoint hash mismatch")

    summary_ref = proof.get("training_summary")
    if not isinstance(summary_ref, dict):
        raise ValueError("candidate proof does not bind a training summary")
    summary_path = Path(summary_ref.get("path", "")).resolve()
    if not summary_path.is_file() or _sha256_json(summary_path) != summary_ref.get(
        "sha256"
    ):
        raise ValueError("candidate proof training summary file/hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "v045_mjlab_training_summary_v2":
        raise ValueError("candidate proof training summary schema mismatch")
    if not summary.get("run_complete", False) or not summary.get(
        "formal_training_run", False
    ):
        raise ValueError("candidate proof training run is not complete and formal")
    watchdog = summary.get("stability_watchdog")
    required_zero_counts = (
        "warning_environment_substeps",
        "abort_environment_substeps",
        "nonfinite_environment_substeps",
        "joint_limit_violation_environment_substeps",
        "constraint_capacity_near_miss_environment_substeps",
        "persistent_event_files",
    )
    if (
        not isinstance(watchdog, dict)
        or watchdog.get("true_physics_substep_hook") is not True
        or watchdog.get("completed_cleanly_with_zero_events") is not True
        or int(watchdog.get("physics_substeps_checked", 0)) <= 0
        or not isinstance(watchdog.get("counts"), dict)
        or any(
            watchdog["counts"].get(name) != 0 for name in required_zero_counts
        )
    ):
        raise ValueError("candidate proof training summary lacks clean watchdog evidence")
    if (
        summary_ref.get("completed_iterations")
        != summary.get("completed_iterations_from_events")
    ):
        raise ValueError("candidate proof training iteration count mismatch")

    summary_checkpoints = {
        (int(item["iteration"]), item["sha256"])
        for item in summary.get("checkpoints", [])
    }
    result_checkpoints = {
        (int(item["iteration"]), item["checkpoint"]["sha256"])
        for item in proof.get("results", [])
    }
    if not summary_checkpoints or result_checkpoints != summary_checkpoints:
        raise ValueError("candidate proof did not evaluate the sealed checkpoint set")
    ranking = proof.get("ranking")
    results = proof.get("results")
    if (
        not isinstance(ranking, list)
        or not isinstance(results, list)
        or sorted(ranking) != list(range(len(results)))
        or results[ranking[0]].get("checkpoint") != selected
    ):
        raise ValueError("candidate proof ranking or selected entry is invalid")
    if any(
        [item.get("seed") for item in result.get("per_seed", [])]
        != [45100, 45101, 45102]
        for result in results
    ):
        raise ValueError("candidate proof has incomplete per-seed evidence")

    for key in (
        "source_selection_proof",
        "bootstrap_provenance",
        "target_model_audit",
    ):
        reference = summary.get(key)
        if not isinstance(reference, dict):
            raise ValueError(f"training summary is missing {key}")
        referenced_path = Path(reference.get("path", "")).resolve()
        if not referenced_path.is_file() or _sha256_json(
            referenced_path
        ) != reference.get("sha256"):
            raise ValueError(f"training summary reference changed: {key}")
    target_audit = json.loads(
        Path(summary["target_model_audit"]["path"]).read_text(encoding="utf-8")
    )
    if (
        target_audit.get("schema_version")
        != "v045_target_model_semantic_audit_v10"
        or not target_audit.get("target_model_semantics_passed", False)
        or not all(bool(value) for value in target_audit.get("checks", {}).values())
    ):
        raise ValueError("candidate proof target-model audit is invalid")
    for source in target_audit.get("sources", {}).values():
        source_path = Path(source.get("path", "")).resolve()
        if not source_path.is_file() or _sha256_json(source_path) != source.get(
            "sha256"
        ):
            raise ValueError(f"target-model audit source changed: {source_path}")
    source_selection = json.loads(
        Path(summary["source_selection_proof"]["path"]).read_text(encoding="utf-8")
    )
    selected_source = source_selection.get(
        "selected_checkpoint_for_target_adaptation", {}
    )
    if (
        target_audit.get("source_action_semantics", {})
        .get("checkpoint", {})
        .get("sha256")
        != selected_source.get("sha256")
    ):
        raise ValueError("target audit source-action checkpoint mismatch")
    action_audit_ref = target_audit.get("sources", {}).get(
        "source_action_semantics_audit", {}
    )
    action_audit = json.loads(
        Path(action_audit_ref.get("path", "")).read_text(encoding="utf-8")
    )
    if (
        action_audit.get("schema_version")
        != "v045_source_action_semantics_audit_v1"
        or not action_audit.get("action_semantics_confirmed", False)
        or action_audit.get("quality_passed") is not False
    ):
        raise ValueError("source-action semantics evidence is invalid")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_json(path),
        "training_summary": {
            "path": str(summary_path),
            "sha256": _sha256_json(summary_path),
        },
        "selected_checkpoint_sha256": checkpoint_sha,
    }


def _axis_angle(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    return np.concatenate(
        [[math.cos(angle / 2.0)], vector * math.sin(angle / 2.0)]
    )


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _robot_geom_ids(model: mujoco.MjModel, prefix: str) -> list[int]:
    result: list[int] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is not None and name.startswith(prefix):
            result.append(geom_id)
    return result


def _robot_body_names(model: mujoco.MjModel, prefix: str) -> tuple[str, ...]:
    names = []
    for body_id in range(model.nbody):
        full_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        if full_name is not None and full_name.startswith(prefix):
            names.append(full_name.removeprefix(prefix))
    if not names:
        raise RuntimeError("RCSS robot has no named bodies")
    return tuple(names)


def _region(body_name: str) -> str:
    name = body_name.lower()
    if "head" in name or name.endswith("h1") or name.endswith("h2"):
        return "head"
    if "torso" in name or "waist" in name:
        return "torso_or_waist"
    if "hand" in name:
        return "hands"
    if "left_foot" in name or "right_foot" in name:
        return "feet"
    if "shank" in name or "ankle" in name:
        return "lower_legs"
    if "hip" in name:
        return "upper_legs"
    if any(token in name for token in ("al1", "al2", "al3", "ar1", "ar2", "ar3")):
        return "arms"
    return "other"


def _contact_snapshot(simulation: object, agent: object) -> dict[str, Any]:
    model = simulation.mj_model
    data = simulation.mj_data
    prefix = agent.agent_id.prefix
    feet = {"left": False, "right": False}
    by_region = {region: 0.0 for region in CONTACT_REGIONS}
    by_body: dict[str, float] = {}
    wrench = np.zeros(6, dtype=np.float64)
    contact_count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        names: list[str] = []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            body_id = int(model.geom_bodyid[geom_id])
            names.append(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            )
        robot_names = [name for name in names if name.startswith(prefix)]
        if not robot_names:
            continue
        contact_count += 1
        mujoco.mj_contactForce(model, data, index, wrench)
        force = float(np.linalg.norm(wrench[:3]))
        for full_name in robot_names:
            short_name = full_name.removeprefix(prefix)
            lowered = short_name.lower()
            if "left_foot" in lowered:
                feet["left"] = True
            if "right_foot" in lowered:
                feet["right"] = True
            region = _region(short_name)
            by_region[region] = max(by_region[region], force)
            by_body[short_name] = max(by_body.get(short_name, 0.0), force)
    return {
        "feet": feet,
        "contact_count": contact_count,
        "maximum_force_by_region_n": by_region,
        "maximum_force_by_body_n": by_body,
    }


class SafetyRecorder:
    """Collect complete per-step contact, joint and actuator evidence."""

    def __init__(
        self,
        *,
        unsafe_force_n: float,
        command_target_tolerance_rad: float,
        physical_excursion_tolerance_rad: float,
        torque_tolerance_nm: float,
        robot_body_names: tuple[str, ...],
    ) -> None:
        self.unsafe_force_n = unsafe_force_n
        self.command_target_tolerance_rad = command_target_tolerance_rad
        self.physical_excursion_tolerance_rad = physical_excursion_tolerance_rad
        self.torque_tolerance_nm = torque_tolerance_nm
        self.phase_steps = {"settle": 0, "recovery": 0, "walk": 0}
        self.maximum_contact_by_region = {
            region: 0.0 for region in CONTACT_REGIONS
        }
        self.maximum_contact_by_body = {
            name: 0.0 for name in robot_body_names
        }
        self.non_foot_contact_steps = 0
        self.command_target_violation_steps = 0
        self.command_target_violation_names: set[str] = set()
        self.maximum_command_target_overshoot_rad = 0.0
        self.physical_excursion_steps = 0
        self.physical_excursion_names: set[str] = set()
        self.maximum_physical_excursion_rad = 0.0
        self.nonfinite_state = False
        self.torque_joint_samples = 0
        self.torque_saturation_joint_samples = 0
        self.torque_saturated_joint_samples = 0
        self.torque_over_limit_joint_samples = 0
        self.torque_over_limit_names: set[str] = set()
        self.maximum_requested_torque_nm = 0.0
        self.maximum_sent_torque_nm = 0.0
        self.maximum_actual_torque_nm = 0.0
        self.maximum_actual_by_joint = {name: 0.0 for name in T1_JOINT_NAMES}
        self.phase_torque = {
            phase: {
                "samples": 0,
                "saturation_samples": 0,
                "saturated": 0,
                "over_limit": 0,
            }
            for phase in self.phase_steps
        }

    def observe(
        self,
        simulation: object,
        agent: object,
        *,
        phase: str,
        torque_limits: np.ndarray,
        requested_torque: np.ndarray,
        sent_torque: np.ndarray,
        command_targets: np.ndarray,
        saturation_mask: np.ndarray = JOINT_SPACE_SATURATION_MASK,
    ) -> dict[str, Any]:
        self.phase_steps[phase] += 1
        contact = _contact_snapshot(simulation, agent)
        for region, force in contact["maximum_force_by_region_n"].items():
            self.maximum_contact_by_region[region] = max(
                self.maximum_contact_by_region[region], float(force)
            )
        for body, force in contact["maximum_force_by_body_n"].items():
            self.maximum_contact_by_body[body] = max(
                self.maximum_contact_by_body.get(body, 0.0), float(force)
            )
        if any(
            force > 0.0
            for region, force in contact["maximum_force_by_region_n"].items()
            if region != "feet"
        ):
            self.non_foot_contact_steps += 1

        joint_pos, joint_vel = read_joint_state(agent)
        actual_torque = np.asarray(
            [float(joint.qfrc_actuator[0]) for joint in agent._joints],
            dtype=np.float64,
        )
        targets = np.asarray(command_targets, dtype=np.float64)
        arrays = (
            joint_pos,
            joint_vel,
            requested_torque,
            sent_torque,
            actual_torque,
            targets,
        )
        if any(np.asarray(array).shape != (23,) for array in arrays):
            raise RuntimeError("safety recorder received a non-23D vector")
        if not all(np.isfinite(array).all() for array in arrays):
            self.nonfinite_state = True

        command_overshoot = np.maximum(
            np.maximum(
                T1_SAFE_HARD_LOWER_LIMITS - targets,
                targets - T1_SAFE_HARD_UPPER_LIMITS,
            ),
            0.0,
        )
        command_violated = command_overshoot > self.command_target_tolerance_rad
        if np.any(command_violated):
            self.command_target_violation_steps += 1
            self.command_target_violation_names.update(
                name
                for name, bad in zip(
                    T1_JOINT_NAMES, command_violated, strict=True
                )
                if bad
            )
        self.maximum_command_target_overshoot_rad = max(
            self.maximum_command_target_overshoot_rad,
            float(np.max(command_overshoot)),
        )
        physical_overshoot = np.maximum(
            np.maximum(
                T1_SAFE_HARD_LOWER_LIMITS - joint_pos,
                joint_pos - T1_SAFE_HARD_UPPER_LIMITS,
            ),
            0.0,
        )
        physical_excursion = (
            physical_overshoot > self.physical_excursion_tolerance_rad
        )
        if np.any(physical_excursion):
            self.physical_excursion_steps += 1
            self.physical_excursion_names.update(
                name
                for name, bad in zip(
                    T1_JOINT_NAMES, physical_excursion, strict=True
                )
                if bad
            )
        self.maximum_physical_excursion_rad = max(
            self.maximum_physical_excursion_rad,
            float(np.max(physical_overshoot)),
        )

        requested = np.asarray(requested_torque, dtype=np.float64)
        sent = np.asarray(sent_torque, dtype=np.float64)
        limits = np.asarray(torque_limits, dtype=np.float64)
        saturation_scope = np.asarray(saturation_mask, dtype=bool)
        if saturation_scope.shape != (23,):
            raise RuntimeError("saturation mask must be one 23D vector")
        # The T1 ankle pitch/roll joints are coupled by a serial-parallel
        # mechanism. Independent joint-space Kp*error is not their actuator
        # request and therefore cannot be used as saturation evidence. Actual
        # qfrc_actuator over-limit checks still cover every one of the 23 joints.
        saturated = (np.abs(requested) >= limits * 0.999) & saturation_scope
        over_limit = np.abs(actual_torque) > limits + self.torque_tolerance_nm
        self.torque_joint_samples += 23
        saturation_samples = int(np.sum(saturation_scope))
        self.torque_saturation_joint_samples += saturation_samples
        self.torque_saturated_joint_samples += int(np.sum(saturated))
        self.torque_over_limit_joint_samples += int(np.sum(over_limit))
        self.phase_torque[phase]["samples"] += 23
        self.phase_torque[phase]["saturation_samples"] += saturation_samples
        self.phase_torque[phase]["saturated"] += int(np.sum(saturated))
        self.phase_torque[phase]["over_limit"] += int(np.sum(over_limit))
        self.torque_over_limit_names.update(
            name for name, bad in zip(T1_JOINT_NAMES, over_limit, strict=True) if bad
        )
        self.maximum_requested_torque_nm = max(
            self.maximum_requested_torque_nm, float(np.max(np.abs(requested)))
        )
        self.maximum_sent_torque_nm = max(
            self.maximum_sent_torque_nm, float(np.max(np.abs(sent)))
        )
        self.maximum_actual_torque_nm = max(
            self.maximum_actual_torque_nm, float(np.max(np.abs(actual_torque)))
        )
        for name, value in zip(T1_JOINT_NAMES, actual_torque, strict=True):
            self.maximum_actual_by_joint[name] = max(
                self.maximum_actual_by_joint[name], float(abs(value))
            )
        return contact

    def result(self) -> dict[str, Any]:
        total_steps = sum(self.phase_steps.values())
        unsafe_regions = sorted(
            region
            for region, force in self.maximum_contact_by_region.items()
            if force > self.unsafe_force_n
        )
        unsafe_bodies = sorted(
            body
            for body, force in self.maximum_contact_by_body.items()
            if force > self.unsafe_force_n
        )
        phase_torque = {}
        for phase, values in self.phase_torque.items():
            samples = int(values["samples"])
            saturation_samples = int(values.get("saturation_samples", samples))
            phase_torque[phase] = {
                **values,
                "saturation_fraction": (
                    float(values["saturated"] / saturation_samples)
                    if saturation_samples
                    else 0.0
                ),
                "actual_over_limit_fraction": (
                    float(values["over_limit"] / samples) if samples else 0.0
                ),
            }
        return {
            "phase_physics_steps": self.phase_steps,
            "total_physics_steps": total_steps,
            "contact": {
                "maximum_force_by_region_n": self.maximum_contact_by_region,
                "maximum_force_by_body_n": dict(
                    sorted(self.maximum_contact_by_body.items())
                ),
                "non_foot_contact_step_fraction": (
                    float(self.non_foot_contact_steps / total_steps)
                    if total_steps
                    else 0.0
                ),
                "unsafe_regions": unsafe_regions,
                "unsafe_bodies": unsafe_bodies,
                "unsafe_any_body_contact": bool(unsafe_regions),
                "unsafe_head_contact": "head" in unsafe_regions,
                "unsafe_hand_or_torso_contact": bool(
                    {"hands", "torso_or_waist"}.intersection(unsafe_regions)
                ),
            },
            "joint": {
                "command_target_limit_violation": (
                    self.command_target_violation_steps > 0
                ),
                "command_target_violation_step_count": (
                    self.command_target_violation_steps
                ),
                "command_target_violation_names": sorted(
                    self.command_target_violation_names
                ),
                "maximum_command_target_overshoot_rad": (
                    self.maximum_command_target_overshoot_rad
                ),
                "physical_constraint_excursion": self.physical_excursion_steps > 0,
                "physical_excursion_step_count": self.physical_excursion_steps,
                "physical_excursion_names": sorted(self.physical_excursion_names),
                "maximum_physical_excursion_rad": self.maximum_physical_excursion_rad,
                "physical_excursion_tolerance_rad": (
                    self.physical_excursion_tolerance_rad
                ),
            },
            "torque": {
                "joint_samples": self.torque_joint_samples,
                "saturation_joint_samples": self.torque_saturation_joint_samples,
                "saturation_scope": "independent_non_parallel_joint_space_requests",
                "saturation_excluded_joint_names": [
                    T1_JOINT_NAMES[index]
                    for index in sorted(PARALLEL_MECH_INDEXES)
                ],
                "saturated_joint_samples": self.torque_saturated_joint_samples,
                "saturation_fraction": (
                    float(
                        self.torque_saturated_joint_samples
                        / self.torque_saturation_joint_samples
                    )
                    if self.torque_saturation_joint_samples
                    else 1.0
                ),
                "actual_over_limit_joint_samples": self.torque_over_limit_joint_samples,
                "actual_over_limit_fraction": (
                    float(
                        self.torque_over_limit_joint_samples
                        / self.torque_joint_samples
                    )
                    if self.torque_joint_samples
                    else 1.0
                ),
                "actual_over_limit_names": sorted(self.torque_over_limit_names),
                "maximum_requested_nm": self.maximum_requested_torque_nm,
                "maximum_sent_nm": self.maximum_sent_torque_nm,
                "maximum_actual_nm": self.maximum_actual_torque_nm,
                "maximum_actual_by_joint_nm": self.maximum_actual_by_joint,
                "by_phase": phase_torque,
            },
            "nonfinite_state": self.nonfinite_state,
        }


def _place_pose(
    simulation: object,
    agent: object,
    pose: str,
    rng: np.random.Generator,
    joint_tolerance_rad: float,
    tilt_deg: int,
) -> dict[str, Any]:
    axis, full_angle = POSES[pose]
    angle = math.copysign(math.radians(tilt_deg), full_angle)
    base = _axis_angle(axis, angle)
    yaw = _axis_angle((0.0, 0.0, 1.0), float(rng.uniform(-math.pi, math.pi)))
    jitter_x = _axis_angle((1.0, 0.0, 0.0), float(rng.uniform(-0.04, 0.04)))
    jitter_y = _axis_angle((0.0, 1.0, 0.0), float(rng.uniform(-0.04, 0.04)))
    quat = _quat_mul(yaw, _quat_mul(jitter_y, _quat_mul(jitter_x, base)))
    quat /= np.linalg.norm(quat)
    joints = clamp_t1_targets(
        HOME_QPOS + rng.uniform(-0.035, 0.035, size=23).astype(np.float32)
    ).astype(np.float64)
    agent.place_at((0.0, 0.0, 0.0), tuple(float(value) for value in quat))
    agent.init_joints(pos=joints)
    simulation.ball.drop_at(20.0, 0.0)
    simulation.mj_data.ctrl[:] = 0.0
    mujoco.mj_forward(simulation.mj_model, simulation.mj_data)

    pitch_id = mujoco.mj_name2id(
        simulation.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "pitch"
    )
    from_to = np.zeros(6, dtype=np.float64)
    distances = [
        float(
            mujoco.mj_geomDistance(
                simulation.mj_model,
                simulation.mj_data,
                pitch_id,
                geom_id,
                10.0,
                from_to,
            )
        )
        for geom_id in _robot_geom_ids(
            simulation.mj_model, agent.agent_id.prefix
        )
    ]
    if not distances:
        raise RuntimeError("RCSS robot has no collision geoms")
    root_height = -min(distances) + 0.005
    agent._qpos[2] = root_height
    agent._qvel[:] = 0.0
    simulation.mj_data.ctrl[:] = 0.0
    mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
    joint_pos, _ = read_joint_state(agent)
    return {
        "root_height_m": float(root_height),
        "minimum_geom_clearance_m": 0.005,
        "joint_limits_valid": bool(
            np.all(joint_pos >= T1_LOWER_LIMITS - joint_tolerance_rad)
            and np.all(joint_pos <= T1_UPPER_LIMITS + joint_tolerance_rad)
        ),
        "finite_state": bool(
            np.isfinite(simulation.mj_data.qpos).all()
            and np.isfinite(simulation.mj_data.qvel).all()
        ),
        "root_quaternion_wxyz": [float(value) for value in quat],
        "initial_joint_position_rad": [float(value) for value in joint_pos],
    }


def _walk_torques(
    targets: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    requested = COMMON_KP * (targets - joint_pos) - COMMON_KD * joint_vel
    sent = requested.copy()
    for index in PARALLEL_MECH_INDEXES:
        position_torque = np.clip(
            (targets[index] - joint_pos[index]) * COMMON_KP[index],
            -TORQUE_LIMIT[index],
            TORQUE_LIMIT[index],
        )
        sent[index] = position_torque - COMMON_KD[index] * joint_vel[index]
    return requested.astype(np.float64), sent.astype(np.float64)


def _episode(
    *,
    checkpoint: Path,
    locomotion_model: Path,
    pose: str,
    seed: int,
    settle_s: float,
    recovery_timeout_s: float,
    max_target_delta_rad: float,
    tilt_deg: int,
    action_source: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    success_cfg = config["success_definition"]
    safety_cfg = config["safety_thresholds"]
    quality_cfg = config["quality_gates"]
    rng = np.random.default_rng(seed)
    simulation, agent = make_single_t1_simulation()
    getup = RcssMjlabGetupController(
        checkpoint,
        effort_limits=RCSS_EFFECTIVE_EFFORT_LIMIT,
        max_target_delta_rad=max_target_delta_rad,
    )
    locomotion = RcssT1Controller(locomotion_model)
    initial = _place_pose(
        simulation,
        agent,
        pose,
        rng,
        float(safety_cfg["command_target_joint_limit_tolerance_rad"]),
        tilt_deg,
    )
    joint_pos, _ = read_joint_state(agent)
    getup.reset(joint_pos)
    dt = float(simulation.timestep)
    substeps = max(1, round(POLICY_DT / dt))
    safety = SafetyRecorder(
        unsafe_force_n=float(safety_cfg["unsafe_contact_force_n"]),
        command_target_tolerance_rad=float(
            safety_cfg["command_target_joint_limit_tolerance_rad"]
        ),
        physical_excursion_tolerance_rad=float(
            safety_cfg["physical_joint_constraint_excursion_tolerance_rad"]
        ),
        torque_tolerance_nm=float(safety_cfg["torque_limit_tolerance_nm"]),
        robot_body_names=_robot_body_names(
            simulation.mj_model, agent.agent_id.prefix
        ),
    )
    control_error: dict[str, str] | None = None

    for _ in range(max(0, round(settle_s / POLICY_DT))):
        getup.hold_current(agent)
        for _ in range(substeps):
            getup.physics_step(simulation, agent)
            simulation.step([])
            safety.observe(
                simulation,
                agent,
                phase="settle",
                torque_limits=RCSS_EFFECTIVE_EFFORT_LIMIT,
                requested_torque=getup.last_requested_torque,
                sent_torque=getup.last_sent_torque,
                command_targets=getup.target,
            )

    continuous_stand_s = 0.0
    maximum_continuous_stand_s = 0.0
    gate_names = (
        "height",
        "upright",
        "root_speed",
        "left_foot_contact",
        "right_foot_contact",
        "both_feet_contact",
    )
    gate_current_s = {name: 0.0 for name in gate_names}
    gate_maximum_s = {name: 0.0 for name in gate_names}
    gate_true_steps = {name: 0 for name in gate_names}
    recovery_physics_steps = 0
    minimum_root_speed = float("inf")
    maximum_root_speed = 0.0
    recovered = False
    recovery_elapsed_s = 0.0
    maximum_height = float(agent.xpos[2])
    maximum_upright = float(torso_upright(agent))
    for _ in range(max(1, round(recovery_timeout_s / POLICY_DT))):
        try:
            if action_source == "checkpoint":
                getup.policy_step(simulation, agent)
            else:
                joint_pos, joint_vel = read_joint_state(agent)
                if action_source == "zero_delta":
                    requested = joint_pos
                elif action_source == "home_target":
                    requested = HOME_QPOS
                elif action_source == "prepare_target":
                    requested = PREPARE_QPOS
                elif action_source == "prepare_pd":
                    requested = PREPARE_QPOS
                else:
                    raise ValueError(f"unknown engineering action source: {action_source}")
                getup.target[:] = filter_t1_dynamic_targets(
                    requested,
                    joint_pos,
                    joint_vel,
                    max_delta_rad=max_target_delta_rad,
                )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            control_error = {"type": type(error).__name__, "message": str(error)}
            break
        for _ in range(substeps):
            if action_source == "prepare_pd":
                joint_pos, joint_vel = read_joint_state(agent)
                requested_torque = (
                    PREPARE_KP * (PREPARE_QPOS - joint_pos)
                    - PREPARE_KD * joint_vel
                )
                getup.target[:] = clamp_t1_targets(PREPARE_QPOS)
                getup.last_requested_torque[:] = requested_torque
                getup.last_sent_torque[:] = np.clip(
                    requested_torque,
                    -RCSS_EFFECTIVE_EFFORT_LIMIT,
                    RCSS_EFFECTIVE_EFFORT_LIMIT,
                )
                locomotion.prepare_step(simulation, agent)
            else:
                getup.physics_step(simulation, agent)
            simulation.step([])
            recovery_elapsed_s += dt
            contact = safety.observe(
                simulation,
                agent,
                phase="recovery",
                torque_limits=RCSS_EFFECTIVE_EFFORT_LIMIT,
                requested_torque=getup.last_requested_torque,
                sent_torque=getup.last_sent_torque,
                command_targets=getup.target,
            )
            height = float(agent.xpos[2])
            upright = float(torso_upright(agent))
            speed = float(np.linalg.norm(agent._qvel[:3]))
            recovery_physics_steps += 1
            minimum_root_speed = min(minimum_root_speed, speed)
            maximum_root_speed = max(maximum_root_speed, speed)
            maximum_height = max(maximum_height, height)
            maximum_upright = max(maximum_upright, upright)
            gate_values = {
                "height": height >= float(success_cfg["torso_height_m_min"]),
                "upright": upright >= float(success_cfg["upright_score_min"]),
                "root_speed": speed <= float(success_cfg["root_speed_mps_max"]),
                "left_foot_contact": bool(contact["feet"]["left"]),
                "right_foot_contact": bool(contact["feet"]["right"]),
                "both_feet_contact": bool(
                    contact["feet"]["left"] and contact["feet"]["right"]
                ),
            }
            for name, passed in gate_values.items():
                if passed:
                    gate_current_s[name] += dt
                    gate_true_steps[name] += 1
                else:
                    gate_current_s[name] = 0.0
                gate_maximum_s[name] = max(
                    gate_maximum_s[name], gate_current_s[name]
                )
            standing = bool(
                gate_values["height"]
                and gate_values["upright"]
                and gate_values["root_speed"]
                and gate_values["both_feet_contact"]
            )
            continuous_stand_s = continuous_stand_s + dt if standing else 0.0
            maximum_continuous_stand_s = max(
                maximum_continuous_stand_s, continuous_stand_s
            )
            if continuous_stand_s >= float(success_cfg["continuous_standing_s"]) - 0.5 * dt:
                recovered = True
                break
        if recovered:
            break

    walk_duration_s = float(success_cfg["then_walk_s"])
    walk_steps_required = round(walk_duration_s / dt)
    walk_steps_completed = 0
    walk_fell = False
    forward_distance_m = 0.0
    lateral_distance_m = 0.0
    planar_distance_m = 0.0
    minimum_walk_height_m: float | None = None
    minimum_walk_upright: float | None = None
    if recovered and control_error is None:
        start_xy = np.asarray(agent.xpos[:2], dtype=np.float64).copy()
        rotation = np.asarray(agent.xmat, dtype=np.float64).reshape(3, 3)
        forward = rotation[:2, 0].copy()
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        left = np.asarray([-forward[1], forward[0]], dtype=np.float64)
        joint_pos, _ = read_joint_state(agent)
        locomotion.policy.reset()
        locomotion.filtered_targets[:] = joint_pos
        policy_time = 0.0
        command = MotionCommand(
            vx=WALK_COMMAND_VX_MPS,
            vy=0.0,
            yaw_rate=0.0,
            head_yaw=0.0,
            head_pitch=0.28,
        )
        for _ in range(walk_steps_required):
            try:
                before_pos, before_vel = read_joint_state(agent)
                targets = locomotion.command_step(
                    simulation, agent, command, policy_time
                )
                if targets.shape != (23,) or not np.isfinite(targets).all():
                    raise FloatingPointError("locomotion returned an invalid 23D target")
                requested, sent = _walk_torques(targets, before_pos, before_vel)
                simulation.step([])
                safety.observe(
                    simulation,
                    agent,
                    phase="walk",
                    # Normal joints use RCSS native position actuators, whose
                    # force caps are the competition model's effective limits.
                    # TORQUE_LIMIT remains the direct-torque cap for the four
                    # parallel ankle mechanisms.
                    torque_limits=RCSS_EFFECTIVE_EFFORT_LIMIT,
                    requested_torque=requested,
                    sent_torque=sent,
                    command_targets=targets,
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                control_error = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                break
            policy_time += dt
            walk_steps_completed += 1
            height = float(agent.xpos[2])
            upright = float(torso_upright(agent))
            minimum_walk_height_m = (
                height
                if minimum_walk_height_m is None
                else min(minimum_walk_height_m, height)
            )
            minimum_walk_upright = (
                upright
                if minimum_walk_upright is None
                else min(minimum_walk_upright, upright)
            )
            if height < 0.34 or upright < 0.20:
                walk_fell = True
        displacement = np.asarray(agent.xpos[:2], dtype=np.float64) - start_xy
        forward_distance_m = float(np.dot(displacement, forward))
        lateral_distance_m = float(np.dot(displacement, left))
        planar_distance_m = float(np.linalg.norm(displacement))

    safety_result = safety.result()
    walk_completed = walk_steps_completed == walk_steps_required
    walk_success = bool(
        recovered
        and walk_completed
        and not walk_fell
        and forward_distance_m
        >= float(success_cfg["forward_walk_distance_m_min"])
    )
    episode_complete = bool(
        control_error is None
        and safety_result["total_physics_steps"] > 0
        and (not recovered or walk_completed)
    )
    safe = bool(
        not safety_result["joint"]["command_target_limit_violation"]
        and not safety_result["joint"]["physical_constraint_excursion"]
        and not safety_result["nonfinite_state"]
        and not safety_result["contact"]["unsafe_any_body_contact"]
        and safety_result["torque"]["actual_over_limit_fraction"]
        <= float(quality_cfg["actual_torque_over_limit_rate_max"])
        and safety_result["torque"]["saturation_fraction"]
        <= float(quality_cfg["mean_torque_saturation_fraction_max"])
    )
    return {
        "pose": pose,
        "seed": seed,
        "episode_complete": episode_complete,
        "control_error": control_error,
        "initial_pose_contract": initial,
        "recovery": {
            "recovered": recovered,
            "elapsed_s": recovery_elapsed_s,
            "continuous_stand_s": continuous_stand_s,
            "maximum_continuous_stand_s": maximum_continuous_stand_s,
            "maximum_height_m": maximum_height,
            "maximum_upright": maximum_upright,
            "minimum_root_speed_mps": (
                minimum_root_speed if recovery_physics_steps > 0 else None
            ),
            "maximum_root_speed_mps": maximum_root_speed,
            "gate_maximum_continuous_s": gate_maximum_s,
            "gate_true_fraction": {
                name: (
                    gate_true_steps[name] / recovery_physics_steps
                    if recovery_physics_steps > 0
                    else 0.0
                )
                for name in gate_names
            },
        },
        "walk": {
            "evaluated": recovered,
            "command_vx_mps": WALK_COMMAND_VX_MPS,
            "required_duration_s": walk_duration_s,
            "steps_required": walk_steps_required,
            "steps_completed": walk_steps_completed,
            "completed": walk_completed,
            "success": walk_success,
            "fell": walk_fell,
            "forward_distance_m": forward_distance_m,
            "lateral_distance_m": lateral_distance_m,
            "planar_distance_m": planar_distance_m,
            "minimum_height_m": minimum_walk_height_m,
            "minimum_upright": minimum_walk_upright,
        },
        "safety": safety_result,
        "success": bool(
            episode_complete
            and initial["joint_limits_valid"]
            and initial["finite_state"]
            and recovered
            and walk_success
            and safe
        ),
    }


def _failed_episode(pose: str, seed: int, error: Exception) -> dict[str, Any]:
    return {
        "pose": pose,
        "seed": seed,
        "episode_complete": False,
        "control_error": {"type": type(error).__name__, "message": str(error)},
        "initial_pose_contract": {
            "joint_limits_valid": False,
            "finite_state": False,
        },
        "recovery": {"recovered": False},
        "walk": {"evaluated": False, "success": False, "fell": False},
        "safety": {
            "contact": {
                "unsafe_any_body_contact": True,
                "unsafe_head_contact": True,
                "unsafe_hand_or_torso_contact": True,
            },
            "joint": {
                "command_target_limit_violation": True,
                "physical_constraint_excursion": True,
            },
            "torque": {
                "saturation_fraction": 1.0,
                "actual_over_limit_fraction": 1.0,
            },
            "nonfinite_state": True,
        },
        "success": False,
    }


def _mean_flag(episodes: list[dict[str, Any]], getter: Any) -> float:
    return float(np.mean([bool(getter(episode)) for episode in episodes]))


def main() -> int:
    args = _parse_args()
    acceptance_path = args.acceptance_config.resolve()
    checkpoint = args.checkpoint.resolve()
    locomotion_model = args.locomotion_model.resolve()
    if not acceptance_path.is_file():
        raise FileNotFoundError(acceptance_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not locomotion_model.is_file():
        raise FileNotFoundError(locomotion_model)
    config = _load_acceptance(acceptance_path)
    protocol = _resolve_protocol(args, config)
    checkpoint_sha = sha256_file(checkpoint)
    locomotion_sha = sha256_file(locomotion_model)
    acceptance_sha = _sha256_json(acceptance_path)
    seeds = [
        protocol["seed_start"]
        + pose_index * protocol["episodes_per_pose"]
        + episode_index
        for pose_index, _ in enumerate(protocol["poses"])
        for episode_index in range(protocol["episodes_per_pose"])
    ]
    candidate_proof = None
    if protocol["formal"]:
        candidate_proof = _validate_candidate_proof(
            args.candidate_proof.resolve(), checkpoint
        )
    fixed_proof = None
    if protocol["formal"] and args.split == "held_out_test":
        fixed_proof = _validate_fixed_proof(
            args.fixed_proof.resolve(),
            checkpoint_sha256=checkpoint_sha,
            locomotion_sha256=locomotion_sha,
            acceptance_sha256=acceptance_sha,
            candidate_proof_sha256=candidate_proof["sha256"],
            held_out_seeds=seeds,
            config=config,
        )

    label = args.split if protocol["formal"] else "engineering_smoke"
    output = args.output.resolve() if args.output else _next_output(checkpoint, label)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical evidence: {output}")

    episodes: list[dict[str, Any]] = []
    for pose_index, pose in enumerate(protocol["poses"]):
        for episode_index in range(protocol["episodes_per_pose"]):
            seed = (
                protocol["seed_start"]
                + pose_index * protocol["episodes_per_pose"]
                + episode_index
            )
            try:
                episode = _episode(
                    checkpoint=checkpoint,
                    locomotion_model=locomotion_model,
                    pose=pose,
                    seed=seed,
                    settle_s=protocol["settle_s"],
                    recovery_timeout_s=protocol["recovery_timeout_s"],
                    max_target_delta_rad=protocol["max_target_delta_rad"],
                    tilt_deg=protocol["tilt_deg"],
                    action_source=protocol["action_source"],
                    config=config,
                )
            except Exception as error:  # Keep evidence and fail closed.
                episode = _failed_episode(pose, seed, error)
            episodes.append(episode)
            print(
                json.dumps(
                    {
                        "pose": pose,
                        "seed": seed,
                        "complete": episode["episode_complete"],
                        "recovered": episode["recovery"]["recovered"],
                        "walk_success": episode["walk"]["success"],
                        "success": episode["success"],
                        "control_error": episode["control_error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    per_pose: dict[str, Any] = {}
    for pose in POSES:
        selected = [episode for episode in episodes if episode["pose"] == pose]
        per_pose[pose] = {
            "episodes": len(selected),
            "completion_rate": (
                _mean_flag(selected, lambda e: e["episode_complete"])
                if selected
                else 0.0
            ),
            "recovery_rate": (
                _mean_flag(selected, lambda e: e["recovery"]["recovered"])
                if selected
                else 0.0
            ),
            "walk_rate": (
                _mean_flag(selected, lambda e: e["walk"]["success"])
                if selected
                else 0.0
            ),
            "success_rate": (
                _mean_flag(selected, lambda e: e["success"])
                if selected
                else 0.0
            ),
        }

    aggregate = {
        "episodes": len(episodes),
        "episode_completion_rate": _mean_flag(
            episodes, lambda e: e["episode_complete"]
        ),
        "recovery_rate": _mean_flag(
            episodes, lambda e: e["recovery"]["recovered"]
        ),
        "post_recovery_walk_rate": _mean_flag(
            episodes, lambda e: e["walk"]["success"]
        ),
        "success_rate": _mean_flag(episodes, lambda e: e["success"]),
        "fall_rate": _mean_flag(episodes, lambda e: e["walk"]["fell"]),
        "command_target_joint_limit_violation_rate": _mean_flag(
            episodes,
            lambda e: e["safety"]["joint"]["command_target_limit_violation"],
        ),
        "physical_joint_constraint_excursion_rate": _mean_flag(
            episodes,
            lambda e: e["safety"]["joint"]["physical_constraint_excursion"],
        ),
        "nonfinite_state_or_action_rate": _mean_flag(
            episodes,
            lambda e: e["safety"]["nonfinite_state"]
            or e["control_error"] is not None,
        ),
        "unsafe_any_body_contact_rate": _mean_flag(
            episodes,
            lambda e: e["safety"]["contact"]["unsafe_any_body_contact"],
        ),
        "unsafe_head_contact_rate": _mean_flag(
            episodes,
            lambda e: e["safety"]["contact"]["unsafe_head_contact"],
        ),
        "unsafe_hand_or_torso_contact_rate": _mean_flag(
            episodes,
            lambda e: e["safety"]["contact"][
                "unsafe_hand_or_torso_contact"
            ],
        ),
        "actual_torque_over_limit_rate": _mean_flag(
            episodes,
            lambda e: e["safety"]["torque"]["actual_over_limit_fraction"] > 0.0,
        ),
        "mean_torque_saturation_fraction": float(
            np.mean(
                [e["safety"]["torque"]["saturation_fraction"] for e in episodes]
            )
        ),
    }
    q = config["quality_gates"]
    required_pose_set = protocol["poses"] == list(config["required_initial_poses"])
    required_count = protocol["episodes_per_pose"] == int(
        config["required_splits"][args.split]["episodes_per_pose"]
    )
    gates = {
        "formal_protocol": protocol["formal"],
        "formal_selected_candidate_proof": bool(
            not protocol["formal"] or candidate_proof is not None
        ),
        "required_pose_set": required_pose_set,
        "required_episode_count": required_count,
        "all_seeds_unique": len(seeds) == len(set(seeds)),
        "held_out_has_passing_disjoint_fixed_proof": bool(
            args.split == "fixed_validation" or fixed_proof is not None
        ),
        "all_initial_poses_valid": all(
            e["initial_pose_contract"]["joint_limits_valid"]
            and e["initial_pose_contract"]["finite_state"]
            for e in episodes
        ),
        "episode_completion_rate": aggregate["episode_completion_rate"]
        >= float(q["episode_completion_rate_min"]),
        "overall_recovery_rate": aggregate["recovery_rate"]
        >= float(q["overall_recovery_rate_min"]),
        "per_pose_recovery_rate": all(
            metrics["recovery_rate"] >= float(q["per_pose_recovery_rate_min"])
            for metrics in per_pose.values()
        ),
        "post_recovery_walk_rate": aggregate["post_recovery_walk_rate"]
        >= float(q["post_recovery_walk_rate_min"]),
        "overall_success_rate": aggregate["success_rate"]
        >= float(q["overall_success_rate_min"]),
        "per_pose_success_rate": all(
            metrics["success_rate"] >= float(q["per_pose_success_rate_min"])
            for metrics in per_pose.values()
        ),
        "fall_rate": aggregate["fall_rate"] <= float(q["fall_rate_max"]),
        "command_target_joint_limit_violation_rate": aggregate[
            "command_target_joint_limit_violation_rate"
        ]
        <= float(q["command_target_joint_limit_violation_rate_max"]),
        "physical_joint_constraint_excursion_rate": aggregate[
            "physical_joint_constraint_excursion_rate"
        ]
        <= float(q["physical_joint_constraint_excursion_rate_max"]),
        "nonfinite_state_or_action_rate": aggregate[
            "nonfinite_state_or_action_rate"
        ]
        <= float(q["nonfinite_state_or_action_rate_max"]),
        "unsafe_any_body_contact_rate": aggregate[
            "unsafe_any_body_contact_rate"
        ]
        <= float(q["unsafe_any_body_contact_rate_max"]),
        "unsafe_head_contact_rate": aggregate["unsafe_head_contact_rate"]
        <= float(q["unsafe_head_contact_rate_max"]),
        "unsafe_hand_or_torso_contact_rate": aggregate[
            "unsafe_hand_or_torso_contact_rate"
        ]
        <= float(q["unsafe_hand_or_torso_contact_rate_max"]),
        "actual_torque_over_limit_rate": aggregate[
            "actual_torque_over_limit_rate"
        ]
        <= float(q["actual_torque_over_limit_rate_max"]),
        "mean_torque_saturation_fraction": aggregate[
            "mean_torque_saturation_fraction"
        ]
        <= float(q["mean_torque_saturation_fraction_max"]),
    }
    stage_passed = all(gates.values())
    result = {
        "schema_version": "v045_rcss_getup_eval_v3",
        "evaluation_split": args.split if protocol["formal"] else "engineering_smoke",
        "requested_split": args.split,
        "formal_protocol": protocol["formal"],
        "poses": protocol["poses"],
        "seed_start": protocol["seed_start"],
        "episodes_per_pose": protocol["episodes_per_pose"],
        "episode_seed_sha256": _seed_fingerprint(seeds),
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "locomotion_model": {
            "path": str(locomotion_model),
            "sha256": locomotion_sha,
        },
        "acceptance_config": {
            "path": str(acceptance_path),
            "sha256": acceptance_sha,
            "schema_version": config["schema_version"],
        },
        "candidate_selection_proof": candidate_proof,
        "fixed_validation_proof": fixed_proof,
        "settle_s": protocol["settle_s"],
        "recovery_timeout_s": protocol["recovery_timeout_s"],
        "max_target_delta_rad": protocol["max_target_delta_rad"],
        "initial_tilt_deg": protocol["tilt_deg"],
        "action_source": protocol["action_source"],
        "episodes": episodes,
        "per_pose": per_pose,
        "aggregate": aggregate,
        "gates": gates,
        "validation_passed": bool(
            protocol["formal"]
            and args.split == "fixed_validation"
            and stage_passed
        ),
        "quality_passed": bool(
            protocol["formal"]
            and args.split == "held_out_test"
            and stage_passed
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "formal_protocol": protocol["formal"],
                "aggregate": aggregate,
                "validation_passed": result["validation_passed"],
                "quality_passed": result["quality_passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if stage_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
