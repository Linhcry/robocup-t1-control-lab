#!/usr/bin/env python3
"""Fail-closed semantic audit for the exact-RCSS v049 receive task."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _next_output() -> Path:
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = root / f"v049_model_semantic_audit_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v049 semantic audit filename")


def _names(model, mujoco, kind, count: int) -> list[str | None]:
    return [mujoco.mj_id2name(model, kind, index) for index in range(count)]


def _runtime_sources(contract_path: Path) -> dict[str, dict[str, str]]:
    paths = {
        "v049_task": EXPERIMENT_ROOT / "rcss_mjlab_task.py",
        "v049_mdp": EXPERIMENT_ROOT / "mdp.py",
        "v049_evaluator": EXPERIMENT_ROOT / "scripts" / "evaluate_v049.py",
        "v049_audit": EXPERIMENT_ROOT / "scripts" / "audit_v049_model.py",
        "acceptance_contract": contract_path,
    }
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in paths.items()
    }


def main() -> int:
    import mujoco
    from mjlab.scene import Scene
    from mjlab.sensor import ContactSensorCfg

    from experiments.v044_mjlab_t1_getup.rcss_getup_controller import (
        RCSS_EFFECTIVE_EFFORT_LIMIT,
    )
    from experiments.v045_rcss_t1_getup_adaptation.rcss_mjlab_task import RCSS_T1_XML
    from experiments.v049_fullbody_receive import mdp
    from experiments.v049_fullbody_receive.rcss_mjlab_task import (
        DANGEROUS_GROUND_BODIES,
        ILLEGAL_BALL_CONTACT_BODIES,
        LEGAL_LOWER_LEG_BODIES,
        TASK_ID,
        get_rcss_ball_spec,
        rcss_fullbody_receive_env_cfg,
    )
    from locomotion.t1_spec import (
        T1_JOINT_NAMES,
        T1_LOWER_LIMITS,
        T1_UPPER_LIMITS,
        clamp_t1_targets,
    )

    output = _next_output()
    contract_path = EXPERIMENT_ROOT / "configs" / "v049_competitive_acceptance.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    sources = _runtime_sources(contract_path)
    cfg = rcss_fullbody_receive_env_cfg()
    source = mujoco.MjModel.from_xml_path(str(RCSS_T1_XML))
    robot_entity = cfg.scene.entities["robot"].build()
    robot = robot_entity.compile()
    ball = get_rcss_ball_spec().compile()
    combined = Scene(cfg.scene, device="cpu").compile()

    source_bodies = _names(source, mujoco, mujoco.mjtObj.mjOBJ_BODY, source.nbody)[1:]
    robot_bodies = _names(robot, mujoco, mujoco.mjtObj.mjOBJ_BODY, robot.nbody)[1:]
    source_geoms = _names(source, mujoco, mujoco.mjtObj.mjOBJ_GEOM, source.ngeom)
    robot_geoms = _names(robot, mujoco, mujoco.mjtObj.mjOBJ_GEOM, robot.ngeom)
    source_joints = _names(source, mujoco, mujoco.mjtObj.mjOBJ_JOINT, source.njnt)[1:]
    robot_joints = _names(robot, mujoco, mujoco.mjtObj.mjOBJ_JOINT, robot.njnt)[1:]
    source_joints[0] = "Head_yaw"
    robot_joints[0] = "Head_yaw"

    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
    legal_sensor = sensors["legal_lower_leg_ball_contact"]
    illegal_sensor = sensors["illegal_robot_ball_contact"]
    danger_sensor = sensors["dangerous_ground_contact"]
    assert isinstance(legal_sensor, ContactSensorCfg)
    assert isinstance(illegal_sensor, ContactSensorCfg)
    assert isinstance(danger_sensor, ContactSensorCfg)
    combined_sensor_names = _names(
        combined, mujoco, mujoco.mjtObj.mjOBJ_SENSOR, combined.nsensor
    )

    action_cfg = cfg.actions["joint_pos"]
    action_ids, action_names = robot_entity.find_joints_by_actuator_names((".*",))
    canonical_action_names = [_name if _name != "AAHead_yaw" else "Head_yaw" for _name in action_names]
    fixtures = np.stack(
        (
            np.zeros(23, dtype=np.float32),
            np.full(23, 100.0, dtype=np.float32),
            np.full(23, -100.0, dtype=np.float32),
            np.linspace(-4.0, 4.0, 23, dtype=np.float32),
        )
    )
    numpy_clamped = np.stack([clamp_t1_targets(row) for row in fixtures])
    torch_clamped = torch.maximum(
        torch.as_tensor(T1_LOWER_LIMITS).view(1, 23),
        torch.minimum(torch.as_tensor(T1_UPPER_LIMITS).view(1, 23), torch.as_tensor(fixtures)),
    ).numpy()

    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms
    actor_names = tuple(actor_terms)
    critic_names = tuple(critic_terms)
    privilege_names = {name for name in critic_names if name.endswith("_privileged")}
    all_robot_bodies = set(str(name) for name in source_bodies)
    legal_bodies = set(LEGAL_LOWER_LEG_BODIES)
    illegal_bodies = set(ILLEGAL_BALL_CONTACT_BODIES)
    mechanical_caps = np.asarray(source.jnt_actfrcrange[1:, 1], dtype=np.float32)
    update_source = inspect.getsource(mdp.update_receive_state)
    evaluator_source = (EXPERIMENT_ROOT / "scripts" / "evaluate_v049.py").read_text(encoding="utf-8")
    fixed = contract["fixed_validation"]
    held = contract["held_out_test"]
    fixed_seeds = set(range(fixed["seed_start_inclusive"], fixed["seed_end_exclusive"]))
    held_seeds = set(range(held["seed_start_inclusive"], held["seed_end_exclusive"]))

    checks = {
        "source_and_v049_robot_body_count_match": source.nbody == robot.nbody == 25,
        "source_and_v049_robot_geom_count_match": source.ngeom == robot.ngeom == 36,
        "source_and_v049_mass_inertia_match": np.allclose(source.body_mass, robot.body_mass, atol=1.0e-9)
        and np.allclose(source.body_inertia, robot.body_inertia, atol=1.0e-9),
        "source_and_v049_geom_physics_match": source_geoms == robot_geoms
        and np.array_equal(source.geom_type, robot.geom_type)
        and np.allclose(source.geom_size, robot.geom_size, atol=1.0e-9)
        and np.allclose(source.geom_friction, robot.geom_friction, atol=1.0e-9),
        "canonical_23dof_joint_order": tuple(source_joints)
        == tuple(robot_joints)
        == tuple(T1_JOINT_NAMES),
        "one_training_actuator_per_joint": robot.nu == 23,
        "action_order_and_type_are_audited": tuple(action_ids) == tuple(range(23))
        and tuple(canonical_action_names) == tuple(T1_JOINT_NAMES)
        and isinstance(action_cfg, mdp.SafeT1JointPositionActionCfg),
        "gpu_clamp_matches_deployment_contract": np.array_equal(numpy_clamped, torch_clamped),
        "torque_gate_matches_rcss_effective_caps": np.allclose(
            mechanical_caps, RCSS_EFFECTIVE_EFFORT_LIMIT, atol=1.0e-6
        ),
        "ball_matches_rcss_world_physics": ball.njnt == 1
        and int(ball.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
        and ball.ngeom == 1
        and np.isclose(ball.geom_size[0, 0], mdp.BALL_RADIUS_M)
        and np.isclose(ball.body_mass.sum(), mdp.BALL_MASS_KG)
        and np.allclose(ball.geom_friction[0], (0.4, 0.01, 0.01)),
        "legal_and_illegal_body_filters_are_disjoint": not (legal_bodies & illegal_bodies),
        "legal_and_illegal_filters_partition_every_robot_body": legal_bodies | illegal_bodies
        == all_robot_bodies,
        "only_foot_and_shank_bodies_are_legal": legal_bodies
        == {"Shank_Left", "left_foot_link", "Shank_Right", "right_foot_link"},
        "body_mode_sensors_cover_unnamed_collision_geoms": legal_sensor.primary.mode == "body"
        and illegal_sensor.primary.mode == "body"
        and set(legal_sensor.primary.pattern) == legal_bodies
        and set(illegal_sensor.primary.pattern) == illegal_bodies,
        "contact_sensors_use_real_ball_secondary": legal_sensor.secondary is not None
        and legal_sensor.secondary.entity == "ball"
        and legal_sensor.secondary.pattern == "ball"
        and illegal_sensor.secondary is not None
        and illegal_sensor.secondary.entity == "ball"
        and illegal_sensor.secondary.pattern == "ball",
        "dangerous_ground_sensor_is_separate": danger_sensor.primary.mode == "body"
        and set(danger_sensor.primary.pattern) == set(DANGEROUS_GROUND_BODIES)
        and danger_sensor.secondary is not None
        and danger_sensor.secondary.entity is None
        and danger_sensor.secondary.pattern == "terrain",
        "compiled_scene_has_all_three_contact_channels": all(
            any(name and name.startswith(prefix) for name in combined_sensor_names)
            for prefix in (
                "legal_lower_leg_ball_contact_",
                "illegal_robot_ball_contact_",
                "dangerous_ground_contact_",
            )
        ),
        "actor_has_exact_deployable_terms_only": actor_names
        == (
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "last_action",
            "ball_relative_position",
            "ball_relative_velocity",
        )
        and not any("privileged" in name for name in actor_names),
        "actor_ball_estimate_has_noise_and_delay": all(
            actor_terms[name].noise is not None and actor_terms[name].delay_max_lag >= 2
            for name in ("ball_relative_position", "ball_relative_velocity")
        ),
        "critic_is_actor_superset_with_explicit_truth": set(actor_names).issubset(critic_names)
        and privilege_names
        == {
            "base_linear_velocity_privileged",
            "ball_world_state_privileged",
            "scenario_privileged",
            "receive_state_privileged",
        },
        "critic_shared_joint_position_matches_v045_unbiased_semantics": (
            critic_terms["joint_pos"].params == {"biased": False}
        ),
        "success_requires_real_legal_contact": "state.legal_touched" in update_source
        and "first_legal_contact_speed" in update_source,
        "success_requires_speed_reduction": "MINIMUM_SPEED_REDUCTION_FRACTION" in update_source
        and mdp.MINIMUM_SPEED_REDUCTION_FRACTION == 0.50,
        "speed_reduction_is_zero_before_first_legal_contact": all(
            token in update_source
            for token in (
                "reduction = torch.where(",
                "state.legal_touched,",
                "torch.zeros_like(ball_speed)",
            )
        ),
        "success_requires_continuous_one_second_control": "state.hold_steps" in update_source
        and mdp.HOLD_SECONDS == 1.0
        and mdp.CONTROL_HZ == 50.0,
        "success_requires_control_radius": "distance <= CONTROL_RADIUS_M" in update_source
        and mdp.CONTROL_RADIUS_M == 0.55,
        "all_safety_failures_gate_success": all(
            token in update_source
            for token in (
                "illegal_touched",
                "dangerous_contact",
                "fell",
                "joint_violation",
                "torque_violation",
                "nonfinite",
                "missed",
            )
        ),
        "four_stage_speed_angle_landing_curriculum": mdp.STAGE_THRESHOLDS
        == (600 * 32, 1500 * 32, 2600 * 32)
        and contract["curriculum"]["axes"]
        == ["incoming speed", "arrival angle", "physical landing point"],
        "fixed_and_heldout_seeds_are_disjoint": not (fixed_seeds & held_seeds),
        "strict_27_cell_episode_counts": fixed["required_episodes"] == 27 * 6
        and held["required_episodes"] == 27 * 20,
        "anti_fraud_contract_is_fail_closed": all(contract["anti_fraud"].values())
        and contract["promotion"]["current_promotion_allowed"] is False,
        "evaluator_hash_binds_checkpoint_sources_audit_and_fixed_proof": all(
            token in evaluator_source
            for token in (
                "checkpoint_sha",
                "contract_sha",
                "_runtime_sources",
                "_valid_audit",
                "_valid_fixed_proof",
                "_valid_video_proof",
            )
        ),
        "strict_evaluator_keeps_actor_noise_and_delay": (
            "rcss_fullbody_receive_env_cfg(play=False)" in evaluator_source
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = bool(checks) and all(checks.values())
    payload = {
        "schema_version": "v049_model_semantic_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": TASK_ID,
        "rcss_robot_xml": {"path": str(RCSS_T1_XML), "sha256": _sha256(RCSS_T1_XML)},
        "counts": {
            "robot_bodies_excluding_world": robot.nbody - 1,
            "robot_geoms": robot.ngeom,
            "robot_hinge_joints": robot.njnt - 1,
            "robot_actuators": robot.nu,
            "actor_dim": mdp.ACTOR_OBS_DIM,
            "critic_dim": mdp.CRITIC_OBS_DIM,
            "compiled_contact_sensor_channels": combined.nsensor,
        },
        "body_contact_partition": {
            "legal": list(LEGAL_LOWER_LEG_BODIES),
            "illegal": list(ILLEGAL_BALL_CONTACT_BODIES),
        },
        "sources": sources,
        "checks": checks,
        "model_semantics_passed": passed,
        "training_completed": False,
        "fixed_validation_passed": False,
        "heldout_validation_passed": False,
        "quality_passed": False,
        "failed_checks": [name for name, value in checks.items() if not value],
        "conclusion": (
            "The v049 physical receive task is structurally valid; no learned receiving ability is claimed."
            if passed
            else "v049 semantic audit failed; training is blocked."
        ),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "model_semantics_passed": passed, "quality_passed": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
