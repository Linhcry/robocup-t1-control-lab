#!/usr/bin/env python3
"""Fail-closed structural and acceptance-contract audit for v047."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_output() -> Path:
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = root / f"v047_target_contract_audit_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v047 audit filename remains")


def _name(model, mujoco, kind, index: int) -> str | None:
    return mujoco.mj_id2name(model, kind, index)


def main() -> int:
    import mujoco
    from mjlab.scene import Scene

    from experiments.v045_rcss_t1_getup_adaptation.rcss_mjlab_task import (
        RCSS_T1_XML,
        get_rcss_t1_robot_cfg,
    )
    from experiments.v047_fullbody_goalkeeper import mdp
    from experiments.v047_fullbody_goalkeeper.rcss_goalkeeper_task import (
        STAGE_THRESHOLDS,
        get_rcss_ball_spec,
        rcss_fullbody_goalkeeper_env_cfg,
        rcss_fullbody_goalkeeper_ppo_runner_cfg,
    )
    from experiments.v047_fullbody_goalkeeper import warmstart
    from experiments.v047_fullbody_goalkeeper.scripts.train_v047 import (
        FORMAL_RESUME_SUPPORTED,
    )
    from locomotion.t1_spec import T1_JOINT_NAMES

    contract_path = EXPERIMENT_ROOT / "configs" / "v047_competitive_acceptance.json"
    plan_path = EXPERIMENT_ROOT / "configs" / "v047_training_plan.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    source_model = mujoco.MjModel.from_xml_path(str(RCSS_T1_XML))
    entity = get_rcss_t1_robot_cfg().build()
    training_model = entity.compile()
    ball_model = get_rcss_ball_spec().compile()

    hinge_names = []
    for joint_id in range(training_model.njnt):
        if int(training_model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        name = _name(training_model, mujoco, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        hinge_names.append("Head_yaw" if name == "AAHead_yaw" else name)
    action_ids, action_names_raw = entity.find_joints_by_actuator_names((".*",))
    action_names = [
        "Head_yaw" if name == "AAHead_yaw" else name for name in action_names_raw
    ]
    robot_mass = float(training_model.body_mass[1:].sum())

    ball_geom_id = mujoco.mj_name2id(
        ball_model, mujoco.mjtObj.mjOBJ_GEOM, "ball"
    )
    ball_body_id = int(ball_model.geom_bodyid[ball_geom_id])
    fixed = contract["fixed_validation"]
    held = contract["held_out_test"]
    fixed_seeds = set(range(fixed["seed_start_inclusive"], fixed["seed_end_exclusive"]))
    held_seeds = set(range(held["seed_start_inclusive"], held["seed_end_exclusive"]))
    cfg = rcss_fullbody_goalkeeper_env_cfg()
    runner_cfg = rcss_fullbody_goalkeeper_ppo_runner_cfg()
    full_scene_model = Scene(cfg.scene, device="cpu").compile()
    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms
    sensor_names = {sensor.name for sensor in cfg.scene.sensors}
    expected_sensors = {
        "ball_foot_calf_contact",
        "ball_legal_body_contact",
        "ball_head_contact",
        "ball_hand_forearm_contact",
        "head_ground_contact",
        "hand_ground_contact",
    }
    checks = {
        "source_rcss_t1_xml_exists": RCSS_T1_XML.is_file(),
        "exact_rcss_training_robot_has_23_hinges": len(hinge_names) == 23,
        "training_joint_order_is_canonical": tuple(hinge_names) == tuple(T1_JOINT_NAMES),
        "policy_action_order_is_canonical": tuple(action_names) == tuple(T1_JOINT_NAMES),
        "policy_action_ids_are_0_through_22": tuple(action_ids) == tuple(range(23)),
        "training_robot_has_36_exact_rcss_geoms": training_model.ngeom == 36,
        "training_robot_mass_is_exact_rcss": abs(robot_mass - 31.614357) <= 1.0e-9,
        "actor_observation_contract_is_87": mdp.ACTOR_OBS_DIM == 87,
        "critic_observation_contract_is_107": mdp.CRITIC_OBS_DIM == 107,
        "action_contract_is_23": mdp.ACTION_DIM == 23,
        "v045_shared_actor_term_order_is_exact": tuple(actor_terms)[:5]
        == (
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "last_action",
        ),
        "v045_shared_actor_joint_velocity_is_not_rescaled": actor_terms[
            "joint_vel"
        ].scale
        is None,
        "v045_shared_actor_joint_velocity_noise_matches": (
            actor_terms["joint_vel"].noise.n_min == -1.5
            and actor_terms["joint_vel"].noise.n_max == 1.5
        ),
        "v045_shared_actor_joint_position_bias_matches": actor_terms[
            "joint_pos"
        ].params
        == {"biased": True},
        "v045_shared_critic_joint_position_is_unbiased": critic_terms[
            "joint_pos"
        ].params
        == {},
        "v045_actor_normalization_contract_matches": runner_cfg.actor.obs_normalization
        is False,
        "v045_critic_normalization_contract_matches": runner_cfg.critic.obs_normalization
        is True,
        "v045_action_scale_contract_matches": abs(
            float(cfg.actions["joint_pos"].scale) - 0.6
        )
        <= 1.0e-12,
        "warmstart_actor_dimension_mapping_is_75_to_87": (
            warmstart.SOURCE_ACTOR_DIM == 75
            and warmstart.TARGET_ACTOR_DIM == 87
        ),
        "warmstart_critic_dimension_mapping_is_78_to_107": (
            warmstart.SOURCE_CRITIC_DIM == 78
            and warmstart.TARGET_CRITIC_DIM == 107
        ),
        "warmstart_base_lin_mapping_is_75_78_to_87_90": (
            warmstart.SOURCE_BASE_LIN_SLICE == (75, 78)
            and warmstart.TARGET_BASE_LIN_SLICE == (87, 90)
        ),
        "warmstart_transform_version_is_v2": warmstart.TRANSFORM_VERSION
        == "v047_v045_shape_aware_warmstart_v2",
        "formal_resume_is_fail_closed_without_lineage_proof": (
            FORMAL_RESUME_SUPPORTED is False
        ),
        "batched_action_uses_canonical_clamper": cfg.actions[
            "joint_pos"
        ].__class__.__name__ == "T1ClampedJointPositionActionCfg",
        "all_strict_contact_sensors_exist": sensor_names == expected_sensors,
        "four_stage_thresholds_match_contract": tuple(STAGE_THRESHOLDS)
        == (19200, 48000, 83200),
        "ball_radius_matches_rcss": np.allclose(
            ball_model.geom_size[ball_geom_id], [0.11, 0.0, 0.0], atol=1.0e-12
        ),
        "ball_mass_matches_rcss": abs(float(ball_model.body_mass[ball_body_id]) - 0.41)
        <= 1.0e-12,
        "ball_friction_matches_rcss": np.allclose(
            ball_model.geom_friction[ball_geom_id], [0.4, 0.01, 0.01], atol=1.0e-12
        ),
        "ball_contact_dimension_matches_rcss": int(ball_model.geom_condim[ball_geom_id])
        == 6,
        "ball_priority_matches_rcss": int(ball_model.geom_priority[ball_geom_id]) == 1,
        "ball_solref_matches_rcss": np.allclose(
            ball_model.geom_solref[ball_geom_id], [-5000.0, -20.0], atol=1.0e-9
        ),
        "full_scene_air_density_matches_rcss": abs(
            float(full_scene_model.opt.density) - 1.2
        )
        <= 1.0e-12,
        "full_scene_air_viscosity_matches_rcss": abs(
            float(full_scene_model.opt.viscosity) - 0.00002
        )
        <= 1.0e-12,
        "configured_physics_timestep_is_200hz": abs(
            float(cfg.sim.mujoco.timestep) - 0.005
        )
        <= 1.0e-12,
        "policy_decimation_is_4_for_50hz": cfg.decimation == 4,
        "fixed_seed_count_is_120": len(fixed_seeds) == fixed["required_episodes"] == 120,
        "heldout_seed_count_is_360": len(held_seeds)
        == held["required_episodes"]
        == 360,
        "fixed_and_heldout_seeds_are_disjoint": fixed_seeds.isdisjoint(held_seeds),
        "heldout_grid_has_27_cells": (
            len(held["lateral_bins_m"]) - 1
        )
        * (len(held["speed_bins_mps"]) - 1)
        * (len(held["height_bins_m"]) - 1)
        == 27,
        "heldout_grid_can_allocate_at_least_8_per_cell": held[
            "required_episodes"
        ]
        // 27
        >= held["minimum_samples_per_lateral_speed_height_cell"]
        == 8,
        "contract_initial_quality_is_false": contract["current_evidence"][
            "quality_passed"
        ]
        is False,
        "training_plan_initial_quality_is_false": plan["quality_passed"] is False,
    }
    passed = bool(checks) and all(checks.values())
    output = _unique_output()
    payload = {
        "schema_version": "v047_target_contract_audit_v1",
        "sources": {
            "rcss_t1_xml": {"path": str(RCSS_T1_XML), "sha256": _sha256(RCSS_T1_XML)},
            "v047_task": {
                "path": str(EXPERIMENT_ROOT / "rcss_goalkeeper_task.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "rcss_goalkeeper_task.py"),
            },
            "v047_mdp": {
                "path": str(EXPERIMENT_ROOT / "mdp.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "mdp.py"),
            },
            "v047_evaluator": {
                "path": str(EXPERIMENT_ROOT / "scripts" / "evaluate_v047.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "scripts" / "evaluate_v047.py"),
            },
            "v047_audit": {
                "path": str(EXPERIMENT_ROOT / "scripts" / "audit_v047_task.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "scripts" / "audit_v047_task.py"),
            },
            "v047_warmstart": {
                "path": str(EXPERIMENT_ROOT / "warmstart.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "warmstart.py"),
            },
            "v047_training_entry": {
                "path": str(EXPERIMENT_ROOT / "scripts" / "train_v047.py"),
                "sha256": _sha256(
                    EXPERIMENT_ROOT / "scripts" / "train_v047.py"
                ),
            },
            "acceptance_contract": {
                "path": str(contract_path),
                "sha256": _sha256(contract_path),
            },
        },
        "counts": {
            "hinge_joints": len(hinge_names),
            "policy_actions": len(action_ids),
            "robot_geoms": training_model.ngeom,
            "contact_sensors": len(sensor_names),
            "fixed_seeds": len(fixed_seeds),
            "heldout_seeds": len(held_seeds),
        },
        "robot_mass_kg": robot_mass,
        "joint_order": hinge_names,
        "action_order": action_names,
        "checks": checks,
        "audit_passed": passed,
        "gpu_parallel_probe_completed": False,
        "long_training_completed": False,
        "fixed_validation_completed": False,
        "held_out_test_completed": False,
        "deployment_ready": False,
        "quality_passed": False,
        "conclusion": (
            "Structure and fail-closed contract passed; no trained goalkeeper exists yet."
            if passed
            else "v047 structure or contract failed; training is blocked."
        ),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "audit_passed": passed,
                "failed_checks": [name for name, ok in checks.items() if not ok],
                "quality_passed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
