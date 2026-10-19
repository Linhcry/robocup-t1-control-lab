"""Auditable solver, COM-randomization and reset profiles for v045."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from experiments.v045_rcss_t1_getup_adaptation import reset_mdp


PROFILE_SCHEMA_VERSION = "v045_training_profile_v1"
DEFAULT_SOLVER_PROFILE = "rcss_mjb"
DEFAULT_BASE_COM_PROFILE = "off"
DEFAULT_RESET_PROFILE = "rcss_eval_v2"
RESET_CALIBRATION_SCHEMA = "v045_reset_height_calibration_v1"
RESET_CALIBRATION_PATH = (
    Path(__file__).resolve().parent
    / "metrics"
    / "v045_reset_height_calibration_run_01.json"
)
PASSIVE_LIMIT_PROBE_SCHEMA = "v045_passive_hold_limit_probe_v1"
PASSIVE_LIMIT_PROBE_PATH = (
    Path(__file__).resolve().parent
    / "metrics"
    / "v045_passive_hold_reset_limit_probe_run_02.json"
)
TILT_CALIBRATION_SCHEMA = "v045_tilt_curriculum_height_calibration_v1"
TILT_CALIBRATION_PATH = (
    Path(__file__).resolve().parent
    / "metrics"
    / "v045_tilt_curriculum_height_calibration_run_01.json"
)
DENSE_TILT_CALIBRATION_SCHEMA = "v045_dense_tilt_height_calibration_v1"
DENSE_TILT_CALIBRATION_PATH = (
    Path(__file__).resolve().parent
    / "metrics"
    / "v045_dense_tilt_height_calibration_run_03.json"
)

SOLVER_PROFILES: dict[str, dict[str, Any]] = {
    "current_mjlab": {
        "timestep": 0.005,
        "integrator": "implicitfast",
        "impratio": 10,
        "cone": "elliptic",
        "jacobian": "auto",
        "solver": "newton",
        "iterations": 10,
        "tolerance": 1.0e-8,
        "ls_iterations": 20,
        "ls_tolerance": 0.01,
        "ccd_iterations": 50,
    },
    "rcss_mjb": {
        "timestep": 0.005,
        "integrator": "euler",
        "impratio": 1,
        "cone": "pyramidal",
        "jacobian": "auto",
        "solver": "newton",
        "iterations": 100,
        "tolerance": 1.0e-8,
        "ls_iterations": 50,
        "ls_tolerance": 0.01,
        "ccd_iterations": 35,
    },
}

BASE_COM_PROFILES: dict[str, dict[str, Any] | None] = {
    "off": None,
    "current": {
        "operation": "add",
        "ranges": {
            0: (-0.025, 0.025),
            1: (-0.025, 0.025),
            2: (-0.03, 0.03),
        },
    },
}

RESET_PROFILES: dict[str, dict[str, Any]] = {
    # Non-promotable first stage of the reverse curriculum.  Every episode
    # begins from the exact RCSS home pose so PPO can learn the terminal skill
    # (quiet two-foot balance) before it is asked to recover from a tilt.
    "engineering_stand_v1": {
        "fall_probability": 0.0,
        "orientation_jitter_rad": 0.0,
        "joint_jitter_rad": 0.0,
        "minimum_clearance_m": 0.005,
    },
    "engineering_stand_jitter_v1": {
        "fall_probability": 0.0,
        "orientation_jitter_rad": 0.0,
        "joint_jitter_rad": 0.0,
        "standing_yaw_randomization": True,
        "standing_orientation_jitter_rad": 0.015,
        "standing_joint_jitter_rad": 0.01,
        "minimum_clearance_m": 0.005,
    },
    "engineering_stand_jitter_v2": {
        "fall_probability": 0.0,
        "orientation_jitter_rad": 0.0,
        "joint_jitter_rad": 0.0,
        "standing_yaw_randomization": True,
        "standing_orientation_jitter_rad": 0.04,
        "standing_joint_jitter_rad": 0.035,
        "minimum_clearance_m": 0.005,
    },
    # Full evaluation-width perturbations plus an anti-saturation objective.
    # This stage teaches a quiet equilibrium before the same objective is
    # carried into progressively larger get-up tilts.
    "engineering_stand_jitter_v3": {
        "fall_probability": 0.0,
        "orientation_jitter_rad": 0.0,
        "joint_jitter_rad": 0.0,
        "standing_yaw_randomization": True,
        "standing_orientation_jitter_rad": 0.04,
        "standing_joint_jitter_rad": 0.035,
        "minimum_clearance_m": 0.005,
        "reward_weights": {
            "raw_action_l2": -0.20,
            "processed_delta_l2": -1.0,
            "normalized_joint_velocity_l2": -0.10,
            "torque_saturation": -0.50,
        },
    },
    "engineering_stand_quiet_v4": {
        "fall_probability": 0.0,
        "orientation_jitter_rad": 0.0,
        "joint_jitter_rad": 0.0,
        "standing_yaw_randomization": True,
        "standing_orientation_jitter_rad": 0.04,
        "standing_joint_jitter_rad": 0.035,
        "minimum_clearance_m": 0.005,
        "reward_weights": {
            "raw_action_l2": -2.0,
            "processed_delta_l2": -2.0,
            "normalized_joint_velocity_l2": -0.10,
            "torque_saturation": -1.0,
        },
    },
    # Non-promotable source-to-RCSS bridge.  The 40% standing resets teach the
    # adapted actor the lower-torque RCSS standing equilibrium before a later
    # run returns to the complete four-pose domain.
    "engineering_bridge_v1": {
        "fall_probability": 0.6,
        "orientation_jitter_rad": 0.02,
        "joint_jitter_rad": 0.02,
        "minimum_clearance_m": 0.005,
    },
    "legacy_narrow": {
        "fall_probability": 1.0,
        "orientation_jitter_rad": 0.02,
        "joint_jitter_rad": 0.02,
        "minimum_clearance_m": 0.005,
    },
    **{
        f"engineering_tilt_{tilt_deg}_v1": {
            "fall_probability": 1.0,
            "orientation_jitter_rad": 0.04,
            "joint_jitter_rad": 0.035,
            "tilt_deg": tilt_deg,
            "minimum_clearance_m": 0.005,
        }
        for tilt_deg in (12, 14, 15, 30, 45, 60, 75, 90)
    },
    **{
        f"engineering_tilt_{tilt_deg}_pitch_quiet_v1": {
            "fall_probability": 1.0,
            "orientation_jitter_rad": 0.04,
            "joint_jitter_rad": 0.035,
            "tilt_deg": tilt_deg,
            "pose_indices": (0, 1),
            "minimum_clearance_m": 0.005,
            "reward_weights": {
                "raw_action_l2": -0.20,
                "processed_delta_l2": -1.0,
                "normalized_joint_velocity_l2": -0.10,
                "torque_saturation": -0.50,
            },
        }
        for tilt_deg in (5, 10, 12, 14)
    },
    **{
        f"engineering_tilt_{tilt_deg}_quiet_v1": {
            "fall_probability": 1.0,
            "orientation_jitter_rad": 0.04,
            "joint_jitter_rad": 0.035,
            "tilt_deg": tilt_deg,
            "minimum_clearance_m": 0.005,
            "reward_weights": {
                "raw_action_l2": -0.20,
                "processed_delta_l2": -1.0,
                "normalized_joint_velocity_l2": -0.10,
                "torque_saturation": -0.50,
            },
        }
        for tilt_deg in (12, 14, 15, 30, 45, 60, 75, 90)
    },
    **{
        f"engineering_tilt_{tilt_deg}_contact_jitter_quiet_v1": {
            "fall_probability": 1.0,
            "orientation_jitter_rad": 0.04,
            "joint_jitter_rad": 0.035,
            "tilt_deg": tilt_deg,
            "root_height_drop_jitter_m": 0.010,
            "minimum_clearance_m": 0.005,
            "reward_weights": {
                "raw_action_l2": -0.20,
                "processed_delta_l2": -1.0,
                "normalized_joint_velocity_l2": -0.10,
                "torque_saturation": -0.50,
            },
        }
        for tilt_deg in (14, 15)
    },
    **{
        f"engineering_tilt_{tilt_deg}_{pose_name}_contact_specialist_v1": {
            "fall_probability": 1.0,
            "orientation_jitter_rad": 0.04,
            "joint_jitter_rad": 0.035,
            "tilt_deg": tilt_deg,
            "pose_indices": (pose_index,),
            "root_height_drop_jitter_m": 0.010,
            "minimum_clearance_m": 0.005,
            "reward_weights": {
                "raw_action_l2": -0.20,
                "processed_delta_l2": -1.0,
                "normalized_joint_velocity_l2": -0.10,
                "torque_saturation": -0.50,
            },
        }
        for tilt_deg in (14, 15)
        for pose_index, pose_name in enumerate(reset_mdp.POSE_NAMES)
    },
    "rcss_eval_v2": {
        "fall_probability": 1.0,
        "orientation_jitter_rad": 0.04,
        "joint_jitter_rad": 0.035,
        "minimum_clearance_m": 0.005,
    },
}


def _solver_snapshot(env_cfg: Any) -> dict[str, Any]:
    mujoco_cfg = env_cfg.sim.mujoco
    return {
        key: getattr(mujoco_cfg, key)
        for key in SOLVER_PROFILES[DEFAULT_SOLVER_PROFILE]
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_reset_calibration() -> dict[str, Any]:
    """Bind every training run to the sampled exact-RCSS height calibration."""

    path = RESET_CALIBRATION_PATH.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"v045 reset calibration is missing: {path}")
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != RESET_CALIBRATION_SCHEMA:
        raise ValueError("v045 reset calibration schema mismatch")
    if proof.get("pose_order") != list(reset_mdp.POSE_NAMES):
        raise ValueError("v045 reset calibration pose order changed")
    if proof.get("selected_root_heights_m") != list(reset_mdp.ROOT_HEIGHTS_M):
        raise ValueError("v045 reset heights differ from their calibration")
    if proof.get("samples_per_pose") != 5000 or proof.get("total_samples") != 20000:
        raise ValueError("v045 reset calibration sample count changed")
    domain = proof.get("domain", {})
    if domain.get("orientation_jitter_rad") != [-0.04, 0.04]:
        raise ValueError("v045 reset orientation calibration domain changed")
    if domain.get("joint_jitter_rad") != [-0.035, 0.035]:
        raise ValueError("v045 reset joint calibration domain changed")
    if domain.get("minimum_clearance_m") != 0.005:
        raise ValueError("v045 reset clearance contract changed")
    if proof.get("all_sampled_resets_clear_at_selected_heights") is not True:
        raise ValueError("v045 reset calibration contains clearance failures")
    per_pose = proof.get("per_pose")
    if not isinstance(per_pose, dict) or set(per_pose) != set(reset_mdp.POSE_NAMES):
        raise ValueError("v045 reset calibration per-pose evidence is incomplete")
    if any(
        row.get("sampled_clearance_failures_at_selected_height") != 0
        for row in per_pose.values()
    ):
        raise ValueError("v045 reset calibration did not clear every sample")
    script = proof.get("calibration_script", {})
    script_path = Path(script.get("path", "")).expanduser().resolve()
    if not script_path.is_file() or script.get("sha256") != _sha256(script_path):
        raise ValueError("v045 reset calibration script changed")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": RESET_CALIBRATION_SCHEMA,
        "seed": proof.get("seed"),
        "samples_per_pose": proof.get("samples_per_pose"),
        "selected_root_heights_m": proof.get("selected_root_heights_m"),
        "all_sampled_resets_clear_at_selected_heights": True,
    }


def _validated_passive_limit_probe() -> dict[str, Any]:
    """Bind soft-constraint thresholds to an exact-RCSS no-policy baseline."""

    path = PASSIVE_LIMIT_PROBE_PATH.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"v045 passive limit probe is missing: {path}")
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != PASSIVE_LIMIT_PROBE_SCHEMA:
        raise ValueError("v045 passive limit probe schema mismatch")
    if proof.get("backend") != "rcssservermj_mujoco_3_5":
        raise ValueError("v045 passive limit probe used the wrong backend")
    aggregate = proof.get("aggregate", {})
    if aggregate.get("episode_count") != 80 or aggregate.get("violation_rate") != 1.0:
        raise ValueError("v045 passive limit probe sample contract changed")
    maximum = aggregate.get("maximum_joint_limit_violation_rad")
    if not isinstance(maximum, (int, float)) or not 0.28 <= maximum < 0.30:
        raise ValueError("v045 passive soft-constraint envelope changed")
    script = proof.get("script", {})
    script_path = Path(script.get("path", "")).expanduser().resolve()
    if not script_path.is_file() or script.get("sha256") != _sha256(script_path):
        raise ValueError("v045 passive limit probe script changed")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": PASSIVE_LIMIT_PROBE_SCHEMA,
        "episode_count": 80,
        "violation_rate_at_0_0001_rad": 1.0,
        "maximum_joint_limit_violation_rad": float(maximum),
        "watchdog_warning_envelope_rad": 0.30,
        "watchdog_abort_envelope_rad": 0.40,
    }


def _validated_tilt_calibration() -> dict[str, Any]:
    """Bind engineering tilt stages to exact-RCSS clearance evidence."""

    path = TILT_CALIBRATION_PATH.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"v045 tilt calibration is missing: {path}")
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != TILT_CALIBRATION_SCHEMA:
        raise ValueError("v045 tilt calibration schema mismatch")
    if proof.get("pose_order") != list(reset_mdp.POSE_NAMES):
        raise ValueError("v045 tilt calibration pose order changed")
    if proof.get("tilt_degrees") != [15, 30, 45, 60, 75, 90]:
        raise ValueError("v045 tilt calibration stage order changed")
    if proof.get("samples_per_pose_level") != 1000 or proof.get(
        "total_samples"
    ) != 24000:
        raise ValueError("v045 tilt calibration sample count changed")
    if proof.get("all_sampled_resets_clear_at_selected_heights") is not True:
        raise ValueError("v045 tilt calibration contains clearance failures")
    selected = proof.get("selected_root_heights_m_by_tilt_deg", {})
    for tilt_deg in (15, 30, 45, 60, 75):
        if selected.get(str(tilt_deg)) != list(
            reset_mdp.TILT_ROOT_HEIGHTS_M[tilt_deg]
        ):
            raise ValueError(f"v045 tilt-{tilt_deg} heights changed")
    script = proof.get("calibration_script", {})
    script_path = Path(script.get("path", "")).expanduser().resolve()
    if not script_path.is_file() or script.get("sha256") != _sha256(script_path):
        raise ValueError("v045 tilt calibration script changed")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": TILT_CALIBRATION_SCHEMA,
        "seed": proof.get("seed"),
        "samples_per_pose_level": proof.get("samples_per_pose_level"),
        "all_sampled_resets_clear_at_selected_heights": True,
    }


def _validated_dense_tilt_calibration() -> dict[str, Any]:
    """Bind the 5/10/12/14-degree curriculum to exact collision evidence."""

    path = DENSE_TILT_CALIBRATION_PATH.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"v045 dense-tilt calibration is missing: {path}")
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != DENSE_TILT_CALIBRATION_SCHEMA:
        raise ValueError("v045 dense-tilt calibration schema mismatch")
    if proof.get("pose_order") != list(reset_mdp.POSE_NAMES):
        raise ValueError("v045 dense-tilt calibration pose order changed")
    if proof.get("tilt_degrees") != [5, 10, 12, 14]:
        raise ValueError("v045 dense-tilt calibration stage order changed")
    if proof.get("samples_per_pose_level") != 2000 or proof.get(
        "total_samples"
    ) != 32000:
        raise ValueError("v045 dense-tilt calibration sample count changed")
    if proof.get("all_sampled_resets_clear_at_selected_heights") is not True:
        raise ValueError("v045 dense-tilt calibration contains clearance failures")
    selected = proof.get("selected_root_heights_m_by_tilt_deg", {})
    for tilt_deg in (5, 10, 12, 14):
        if selected.get(str(tilt_deg)) != list(
            reset_mdp.TILT_ROOT_HEIGHTS_M[tilt_deg]
        ):
            raise ValueError(f"v045 dense tilt-{tilt_deg} heights changed")
    references = [proof.get("calibration_script"), *proof.get("dependencies", [])]
    if len(references) != 4:
        raise ValueError("v045 dense-tilt calibration dependencies are incomplete")
    for reference in references:
        if not isinstance(reference, dict):
            raise ValueError("v045 dense-tilt calibration reference is invalid")
        reference_path = Path(reference.get("path", "")).expanduser().resolve()
        if (
            not reference_path.is_file()
            or reference.get("sha256") != _sha256(reference_path)
        ):
            raise ValueError(
                f"v045 fine-tilt calibration dependency changed: {reference_path}"
            )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": DENSE_TILT_CALIBRATION_SCHEMA,
        "seed": proof.get("seed"),
        "samples_per_pose_level": proof.get("samples_per_pose_level"),
        "all_sampled_resets_clear_at_selected_heights": True,
    }


def apply_training_profiles(
    env_cfg: Any,
    *,
    solver_profile: str = DEFAULT_SOLVER_PROFILE,
    base_com_profile: str = DEFAULT_BASE_COM_PROFILE,
    reset_profile: str = DEFAULT_RESET_PROFILE,
) -> dict[str, Any]:
    """Apply one controlled profile and return a JSON-safe audit snapshot."""

    if solver_profile not in SOLVER_PROFILES:
        raise ValueError(f"unknown v045 solver profile: {solver_profile}")
    if base_com_profile not in BASE_COM_PROFILES:
        raise ValueError(f"unknown v045 base COM profile: {base_com_profile}")
    if reset_profile not in RESET_PROFILES:
        raise ValueError(f"unknown v045 reset profile: {reset_profile}")

    for key, value in SOLVER_PROFILES[solver_profile].items():
        setattr(env_cfg.sim.mujoco, key, value)

    if base_com_profile == "off":
        env_cfg.events.pop("base_com", None)
    else:
        if "base_com" not in env_cfg.events:
            env_cfg.events["base_com"] = EventTermCfg(
                func=envs_mdp.body_com_offset,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
                    "operation": "add",
                    "ranges": deepcopy(BASE_COM_PROFILES["current"]["ranges"]),
                },
            )
        event = env_cfg.events["base_com"]
        selected = BASE_COM_PROFILES[base_com_profile]
        assert selected is not None
        event.params["operation"] = selected["operation"]
        event.params["ranges"] = deepcopy(selected["ranges"])

    reset = env_cfg.events["reset_fallen_or_standing"]
    reset.func = reset_mdp.reset_rcss_four_pose
    selected_reset = RESET_PROFILES[reset_profile]
    reset.params = {
        "fall_probability": selected_reset["fall_probability"],
        "orientation_jitter_rad": selected_reset["orientation_jitter_rad"],
        "joint_jitter_rad": selected_reset["joint_jitter_rad"],
    }
    if "tilt_deg" in selected_reset:
        reset.params["tilt_deg"] = int(selected_reset["tilt_deg"])
    if "pose_indices" in selected_reset:
        reset.params["pose_indices"] = tuple(selected_reset["pose_indices"])
    if "root_height_drop_jitter_m" in selected_reset:
        reset.params["root_height_drop_jitter_m"] = float(
            selected_reset["root_height_drop_jitter_m"]
        )
    for name in (
        "standing_yaw_randomization",
        "standing_orientation_jitter_rad",
        "standing_joint_jitter_rad",
    ):
        if name in selected_reset:
            reset.params[name] = selected_reset[name]

    reward_weights = deepcopy(selected_reset.get("reward_weights", {}))
    for reward_name, reward_weight in reward_weights.items():
        if reward_name not in env_cfg.rewards:
            raise KeyError(f"unknown v045 reward override: {reward_name}")
        if not isinstance(reward_weight, (int, float)):
            raise TypeError(f"v045 reward weight must be numeric: {reward_name}")
        env_cfg.rewards[reward_name].weight = float(reward_weight)

    actual_base_com = None
    if "base_com" in env_cfg.events:
        event = env_cfg.events["base_com"]
        actual_base_com = {
            "operation": event.params["operation"],
            "ranges": deepcopy(event.params["ranges"]),
        }

    payload = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "selected": {
            "solver": solver_profile,
            "base_com": base_com_profile,
            "reset": reset_profile,
        },
        "actual": {
            "solver": _solver_snapshot(env_cfg),
            "base_com": actual_base_com,
            "reset": deepcopy(reset.params),
            "reset_root_heights_m": list(
                reset_mdp.TILT_ROOT_HEIGHTS_M[
                    int(selected_reset.get("tilt_deg", 90))
                ]
            ),
            "reset_height_calibration": _validated_reset_calibration(),
            "tilt_height_calibration": (
                _validated_tilt_calibration()
                if int(selected_reset.get("tilt_deg", 0)) in (15, 30, 45, 60, 75, 90)
                else None
            ),
            "dense_tilt_height_calibration": (
                _validated_dense_tilt_calibration()
                if int(selected_reset.get("tilt_deg", 0)) in (5, 10, 12, 14)
                else None
            ),
            "passive_soft_joint_constraint_probe": _validated_passive_limit_probe(),
            "reset_pose_order": list(reset_mdp.POSE_NAMES),
            "minimum_clearance_m": selected_reset["minimum_clearance_m"],
            "decimation": int(env_cfg.decimation),
            "physics_dt_s": float(env_cfg.sim.mujoco.timestep),
            "policy_dt_s": float(env_cfg.decimation * env_cfg.sim.mujoco.timestep),
            "reward_weights": {
                name: float(env_cfg.rewards[name].weight)
                for name in (
                    "raw_action_l2",
                    "processed_delta_l2",
                    "normalized_joint_velocity_l2",
                    "torque_saturation",
                )
                if name in env_cfg.rewards
            },
        },
        "formal_component_requirements": {
            "solver": DEFAULT_SOLVER_PROFILE,
            "reset": DEFAULT_RESET_PROFILE,
            "base_com": "must be selected by immutable controlled-A/B proof",
        },
        "formal_components_match": bool(
            solver_profile == DEFAULT_SOLVER_PROFILE
            and reset_profile == DEFAULT_RESET_PROFILE
        ),
    }
    if payload["actual"]["solver"] != SOLVER_PROFILES[solver_profile]:
        raise RuntimeError("applied v045 solver profile does not match its definition")
    if payload["actual"]["base_com"] != BASE_COM_PROFILES[base_com_profile]:
        raise RuntimeError("applied v045 base COM profile does not match its definition")
    return payload
