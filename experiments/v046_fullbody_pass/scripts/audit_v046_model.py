#!/usr/bin/env python3
"""Fail-closed semantic audit for the v046 robot, ball and task contract."""

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
    metrics = EXPERIMENT_ROOT / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = metrics / f"v046_model_semantic_audit_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v046 audit filename")


def _names(model, mujoco, kind, count: int) -> list[str | None]:
    return [mujoco.mj_id2name(model, kind, index) for index in range(count)]


def main() -> int:
    import mujoco

    from experiments.v045_rcss_t1_getup_adaptation.rcss_mjlab_task import RCSS_T1_XML
    from experiments.v046_fullbody_pass.mdp import rewards as pass_rewards
    from experiments.v046_fullbody_pass.mdp.actions import (
        SafeT1JointPositionActionCfg,
    )
    from experiments.v046_fullbody_pass.mdp.commands import DirectionalPassCommand
    from experiments.v046_fullbody_pass.rcss_mjlab_task import (
        BALL_MASS_KG,
        BALL_RADIUS_M,
        RECEIVE_STAGE_INTERFACE,
        TASK_ID,
        get_soccer_ball_spec,
        rcss_t1_directional_pass_env_cfg,
    )
    from experiments.v046_fullbody_pass import warmstart as pass_warmstart
    from locomotion.t1_spec import (
        T1_JOINT_NAMES,
        T1_LOWER_LIMITS,
        T1_UPPER_LIMITS,
        clamp_t1_targets,
    )
    from locomotion.rcss_t1_controller import TORQUE_LIMIT
    from mjlab.scene import Scene
    from mjlab.sensor import ContactSensorCfg

    output = _next_output()
    cfg = rcss_t1_directional_pass_env_cfg()
    source = mujoco.MjModel.from_xml_path(str(RCSS_T1_XML))
    robot_entity = cfg.scene.entities["robot"].build()
    robot = robot_entity.compile()
    ball = get_soccer_ball_spec().compile()
    scene = Scene(cfg.scene, device="cpu")
    combined = scene.compile()

    source_joint_names = _names(source, mujoco, mujoco.mjtObj.mjOBJ_JOINT, source.njnt)
    robot_joint_names = _names(robot, mujoco, mujoco.mjtObj.mjOBJ_JOINT, robot.njnt)
    source_hinges = source_joint_names[1:]
    robot_hinges = robot_joint_names[1:]
    source_hinges[0] = "Head_yaw"
    robot_hinges[0] = "Head_yaw"
    source_geom_names = _names(source, mujoco, mujoco.mjtObj.mjOBJ_GEOM, source.ngeom)
    robot_geom_names = _names(robot, mujoco, mujoco.mjtObj.mjOBJ_GEOM, robot.ngeom)

    foot_sensor = next(s for s in cfg.scene.sensors if s.name == "foot_ball_contact")
    upper_sensor = next(
        s for s in cfg.scene.sensors if s.name == "upper_body_ball_contact"
    )
    assert isinstance(foot_sensor, ContactSensorCfg)
    assert isinstance(upper_sensor, ContactSensorCfg)
    combined_sensor_names = _names(
        combined, mujoco, mujoco.mjtObj.mjOBJ_SENSOR, combined.nsensor
    )

    action_cfg = cfg.actions["joint_pos"]
    action_ids, action_names = robot_entity.find_joints_by_actuator_names((".*",))
    canonical_action_names = [
        "Head_yaw" if name == "AAHead_yaw" else name for name in action_names
    ]
    fixtures = np.stack(
        [
            np.zeros(23, dtype=np.float32),
            np.full(23, 100.0, dtype=np.float32),
            np.full(23, -100.0, dtype=np.float32),
            np.linspace(-4.0, 4.0, 23, dtype=np.float32),
        ]
    )
    numpy_clamped = np.stack([clamp_t1_targets(row) for row in fixtures])
    torch_clamped = torch.maximum(
        torch.as_tensor(T1_LOWER_LIMITS).view(1, 23),
        torch.minimum(
            torch.as_tensor(T1_UPPER_LIMITS).view(1, 23),
            torch.as_tensor(fixtures),
        ),
    ).numpy()

    actor_names = tuple(cfg.observations["actor"].terms)
    critic_names = tuple(cfg.observations["critic"].terms)
    privileged = tuple(name for name in critic_names if name.endswith("_privileged"))
    required_safety = {
        "upper_body_ball",
        "fallen",
        "hard_joint_limit",
        "hard_torque_limit",
        "ball_out_of_bounds",
    }
    source_success = inspect.getsource(DirectionalPassCommand.success_now)
    reward_success = inspect.getsource(pass_rewards.pass_success_reward)
    contact_found_source = inspect.getsource(pass_rewards.DirectionalPassCommand.current_foot_contact)
    sensor_found_source = inspect.getsource(
        sys.modules[DirectionalPassCommand.__module__]._sensor_found
    )
    acceptance_path = (
        EXPERIMENT_ROOT / "configs" / "v046_competitive_acceptance_v2.json"
    )
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    fixed_cfg = acceptance["required_splits"]["fixed_validation"]
    held_cfg = acceptance["required_splits"]["held_out_test"]
    fixed_seeds = set(
        range(fixed_cfg["seed_start"], fixed_cfg["seed_start"] + fixed_cfg["episodes"])
    )
    held_seeds = set(
        range(held_cfg["seed_start"], held_cfg["seed_start"] + held_cfg["episodes"])
    )
    evaluation_source = (
        EXPERIMENT_ROOT / "scripts" / "evaluate_v046.py"
    ).read_text(encoding="utf-8")

    checks = {
        "source_and_v046_have_24_robot_bodies": source.nbody == robot.nbody == 25,
        "source_and_v046_have_36_robot_geoms": source.ngeom == robot.ngeom == 36,
        "source_and_v046_robot_mass_match": np.isclose(
            source.body_mass.sum(), robot.body_mass.sum(), atol=1.0e-9
        ),
        "source_and_v046_body_mass_inertia_match": np.allclose(
            source.body_mass, robot.body_mass, atol=1.0e-9
        ) and np.allclose(source.body_inertia, robot.body_inertia, atol=1.0e-9),
        "source_and_v046_geom_names_match": source_geom_names == robot_geom_names,
        "source_and_v046_geom_physics_match": (
            np.array_equal(source.geom_type, robot.geom_type)
            and np.allclose(source.geom_size, robot.geom_size, atol=1.0e-9)
            and np.allclose(source.geom_pos, robot.geom_pos, atol=1.0e-9)
            and np.allclose(source.geom_quat, robot.geom_quat, atol=1.0e-9)
            and np.allclose(source.geom_friction, robot.geom_friction, atol=1.0e-9)
        ),
        "joint_order_is_canonical_23": (
            tuple(source_hinges) == tuple(robot_hinges) == tuple(T1_JOINT_NAMES)
        ),
        "source_and_v046_joint_ranges_match": np.allclose(
            source.jnt_range, robot.jnt_range, atol=1.0e-9
        ),
        "v046_has_one_actuator_per_joint": robot.nu == 23,
        "policy_action_order_is_canonical_23": (
            tuple(action_ids) == tuple(range(23))
            and tuple(canonical_action_names) == tuple(T1_JOINT_NAMES)
            and isinstance(action_cfg, SafeT1JointPositionActionCfg)
        ),
        "gpu_clamp_matches_deployment_contract": np.array_equal(
            numpy_clamped, torch_clamped
        ),
        "ball_is_free_0_43kg_0_11m_sphere": (
            ball.njnt == 1
            and int(ball.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
            and ball.ngeom == 1
            and int(ball.geom_type[0]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
            and np.isclose(ball.geom_size[0, 0], BALL_RADIUS_M, atol=1.0e-9)
            and np.isclose(ball.body_mass.sum(), BALL_MASS_KG, atol=1.0e-9)
        ),
        "foot_contact_sensor_is_named_feet_to_ball": (
            tuple(foot_sensor.primary.pattern) == ("left_foot", "right_foot")
            and foot_sensor.primary.entity == "robot"
            and foot_sensor.secondary is not None
            and foot_sensor.secondary.pattern == "ball_geom"
            and foot_sensor.secondary.entity == "ball"
            and set(foot_sensor.fields) >= {"found", "force", "pos"}
        ),
        "upper_contact_sensor_is_separate_and_fail_closed": (
            upper_sensor.primary.entity == "robot"
            and upper_sensor.primary.mode == "body"
            and set(upper_sensor.primary.pattern)
            == {
                "torso",
                "H1",
                "H2",
                "AL1",
                "AL2",
                "AL3",
                "left_hand_link",
                "AR1",
                "AR2",
                "AR3",
                "right_hand_link",
                "Waist",
            }
            and upper_sensor.secondary is not None
            and upper_sensor.secondary.pattern == "ball_geom"
        ),
        "compiled_scene_contains_all_contact_channels": (
            sum(name.startswith("foot_ball_contact_") for name in combined_sensor_names if name)
            == 6
            and sum(
                name.startswith("upper_body_ball_contact_")
                for name in combined_sensor_names
                if name
            )
            == 36
        ),
        "actor_has_only_deployable_terms": (
            actor_names
            == (
                "base_ang_vel",
                "projected_gravity",
                "joint_pos",
                "joint_vel",
                "last_action",
                "ball_relative_position",
                "ball_relative_velocity",
                "target_direction",
                "target_distance",
            )
            and not any("privileged" in name for name in actor_names)
        ),
        "critic_is_actor_superset_with_explicit_privilege": (
            set(actor_names).issubset(critic_names)
            and set(privileged)
            == {
                "base_linear_velocity_privileged",
                "ball_world_state_privileged",
                "contact_state_privileged",
                "pass_progress_privileged",
                "root_height_privileged",
            }
        ),
        "success_is_gated_by_foot_and_rejects_upper_contact": (
            "true_foot_contact_latched" in source_success
            and "failure_contact_latched" in source_success
            and "success_now" in reward_success
        ),
        "contact_latches_cover_all_four_physics_substeps": (
            foot_sensor.history_length == upper_sensor.history_length == cfg.decimation == 4
            and "_sensor_found" in contact_found_source
            and "force_history" in sensor_found_source
        ),
        "strict_safety_terminations_present": required_safety.issubset(
            cfg.terminations
        ),
        "training_and_evaluation_use_deployment_torque_limits": (
            np.array_equal(np.asarray(pass_rewards.EFFORT_LIMITS), TORQUE_LIMIT)
            and "TORQUE_LIMIT" in evaluation_source
            and "RCSS_EFFECTIVE_EFFORT_LIMIT" not in evaluation_source
        ),
        "receive_stage_is_explicitly_unimplemented": (
            RECEIVE_STAGE_INTERFACE == "receive_stage_reserved_not_implemented"
        ),
        "v045_to_v046_mapping_matches_observation_layout": (
            pass_warmstart.V045_ACTOR_DIM == 75
            and pass_warmstart.V046_ACTOR_DIM == 84
            and pass_warmstart.V045_CRITIC_DIM == 78
            and pass_warmstart.V046_CRITIC_DIM == 101
            and pass_warmstart.SHARED_PROPRIO_SLICE == (0, 75)
            and pass_warmstart.V045_BASE_LIN_SLICE == (75, 78)
            and pass_warmstart.V046_BASE_LIN_SLICE == (84, 87)
        ),
        "strict_fixed_and_heldout_seed_sets_are_complete_and_disjoint": (
            len(fixed_seeds) == 80
            and len(held_seeds) == 400
            and not fixed_seeds.intersection(held_seeds)
        ),
        "strict_evaluation_contact_contract_matches_compiled_sensors": (
            tuple(acceptance["episode_contract"]["valid_foot_contact_channels"])
            == tuple(foot_sensor.primary.pattern)
            and tuple(
                acceptance["episode_contract"]["illegal_upper_body_channels"]
            )
            == tuple(upper_sensor.primary.pattern)
        ),
        "strict_evaluator_binds_checkpoint_contract_sources_seeds_and_fixed_proof": all(
            marker in evaluation_source
            for marker in (
                '"checkpoint"',
                '"acceptance_config"',
                '"source_sha256"',
                '"episode_seed_sha256"',
                '"scenario_manifest_sha256"',
                "_validate_fixed_proof",
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = bool(checks) and all(checks.values())
    sources = {}
    for path in sorted(EXPERIMENT_ROOT.rglob("*.py")):
        if "__pycache__" not in path.parts:
            sources[str(path.relative_to(EXPERIMENT_ROOT))] = _sha256(path)
    payload = {
        "schema_version": "v046_model_semantic_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": TASK_ID,
        "rcss_robot_xml": {"path": str(RCSS_T1_XML), "sha256": _sha256(RCSS_T1_XML)},
        "counts": {
            "robot_bodies_excluding_world": robot.nbody - 1,
            "robot_geoms": robot.ngeom,
            "robot_hinge_joints": robot.njnt - 1,
            "robot_actuators": robot.nu,
            "ball_dynamic_bodies": ball.nbody - 1,
            "combined_contact_sensor_channels": combined.nsensor,
        },
        "observation_contract": {
            "actor_terms": actor_names,
            "critic_terms": critic_names,
            "actor_expected_dim": 84,
            "critic_expected_dim": 101,
        },
        "strict_evaluation_contract": {
            "path": str(acceptance_path),
            "sha256": _sha256(acceptance_path),
            "fixed_episodes": fixed_cfg["episodes"],
            "heldout_episodes": held_cfg["episodes"],
            "seed_sets_disjoint": not fixed_seeds.intersection(held_seeds),
        },
        "checks": checks,
        "source_sha256": sources,
        "model_semantics_passed": passed,
        "training_completed": False,
        "validation_passed": False,
        "quality_passed": False,
        "failed_checks": [name for name, value in checks.items() if not value],
        "conclusion": (
            "The exact-RCSS v046 task contract is structurally valid; skill training "
            "and independent evaluation are still required."
            if passed
            else "v046 semantic audit failed; training is blocked."
        ),
    }
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "model_semantics_passed": passed,
                "quality_passed": False,
                "failed_checks": payload["failed_checks"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
