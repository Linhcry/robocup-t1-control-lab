"""Fail-closed structural and real cross-entity contact audit for v048."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v048_fullbody_legal_tackle.rcss_mjlab_task import (  # noqa: E402
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    EFFORT_LIMITS,
    MINIMUM_CLEARANCE_M,
    MINIMUM_CONTROL_SECONDS,
    RCSS_JOINT_NAMES,
    RCSS_T1_XML,
    V048_ACTOR_OBSERVATION_LAYOUT,
    defender_local_observation_components,
    exact_rcss_tackle_env_cfg,
)
from experiments.v048_fullbody_legal_tackle.scripts.evaluate_v048 import (  # noqa: E402
    runtime_control_sources,
)
from experiments.v048_fullbody_legal_tackle.scripts._evidence import write_new_json  # noqa: E402
from locomotion.t1_spec import (  # noqa: E402
    T1_LOWER_LIMITS,
    T1_UPPER_LIMITS,
    clamp_t1_targets,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collision_probe(env) -> dict[str, float | bool]:
    """Overlap torso collision boxes and advance raw physics without auto-reset."""

    attacker = env.scene["attacker"]
    defender = env.scene["defender"]
    ball = env.scene["ball"]
    attacker_state = attacker.data.default_root_state.clone()
    defender_state = defender.data.default_root_state.clone()
    ball_state = ball.data.default_root_state.clone()
    attacker_state[:, :3] = torch.tensor([0.0, 0.0, 0.665], device=env.device)
    defender_state[:, :3] = torch.tensor([0.08, 0.0, 0.665], device=env.device)
    attacker_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    defender_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    attacker_state[:, 7:13] = 0.0
    defender_state[:, 7:13] = 0.0
    ball_state[:, :3] = torch.tensor([3.0, 3.0, 0.11], device=env.device)
    ball_state[:, 7:13] = 0.0
    attacker.write_root_state_to_sim(attacker_state)
    defender.write_root_state_to_sim(defender_state)
    ball.write_root_state_to_sim(ball_state)
    attacker.write_joint_state_to_sim(
        attacker.data.default_joint_pos, attacker.data.default_joint_vel
    )
    defender.write_joint_state_to_sim(
        defender.data.default_joint_pos, defender.data.default_joint_vel
    )
    env.action_manager.process_action(torch.zeros((1, ACTION_DIM), device=env.device))
    env.scene.write_data_to_sim()
    env.sim.forward()
    for _ in range(4):
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
    env.sim.forward()
    env.sim.sense()
    impact = env.scene["cross_robot_impact"].data
    upper_a = env.scene["defender_upper_vs_attacker"].data
    upper_b = env.scene["attacker_upper_vs_defender"].data
    lower_a = env.scene["defender_lower_vs_attacker"].data
    lower_b = env.scene["attacker_lower_vs_defender"].data
    assert impact.found is not None and impact.force is not None
    assert (
        upper_a.found is not None
        and upper_b.found is not None
        and lower_a.found is not None
        and lower_b.found is not None
    )
    return {
        "cross_robot_contact_found": bool((impact.found > 0).any()),
        "cross_robot_max_force_n": float(torch.linalg.vector_norm(impact.force, dim=-1).max()),
        "upper_body_contact_found": bool(
            (upper_a.found > 0).any() or (upper_b.found > 0).any()
        ),
        "lower_body_contact_found": bool(
            (lower_a.found > 0).any() or (lower_b.found > 0).any()
        ),
        "finite_after_impact": bool(
            torch.isfinite(env.sim.data.qpos).all()
            and torch.isfinite(env.sim.data.qvel).all()
            and torch.isfinite(env.sim.data.ctrl).all()
        ),
    }


def _real_transfer_probe(env) -> dict[str, float | bool]:
    """Create a real defender foot-ball contact after the audited attacker touch."""

    attacker = env.scene["attacker"]
    defender = env.scene["defender"]
    ball = env.scene["ball"]
    attacker_state = attacker.data.default_root_state.clone()
    defender_state = defender.data.default_root_state.clone()
    ball_state = ball.data.default_root_state.clone()
    attacker_state[:, :3] = torch.tensor([3.0, 3.0, 0.665], device=env.device)
    defender_state[:, :3] = torch.tensor([0.0, 0.0, 0.665], device=env.device)
    ball_state[:, :3] = torch.tensor([0.12, 0.0, 0.11], device=env.device)
    for state in (attacker_state, defender_state, ball_state):
        state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        state[:, 7:13] = 0.0
    attacker.write_root_state_to_sim(attacker_state)
    defender.write_root_state_to_sim(defender_state)
    ball.write_root_state_to_sim(ball_state)
    attacker.write_joint_state_to_sim(attacker.data.default_joint_pos, attacker.data.default_joint_vel)
    defender.write_joint_state_to_sim(defender.data.default_joint_pos, defender.data.default_joint_vel)
    env._v048_attack_direction[:] = torch.tensor([1.0, 0.0], device=env.device)
    env._v048_attack_speed[:] = 0.0
    env.sim.forward()
    env.step(torch.zeros((1, ACTION_DIM), device=env.device))
    defender_contact = bool(env._v048_defender_touched.all())
    turnover = bool(env._v048_turnover.all() and (env._v048_last_toucher == 2).all())
    control_started = bool((env._v048_max_control_steps >= 1).all())

    ball_state[:, :3] = torch.tensor([-0.19, 0.0, 0.11], device=env.device)
    ball_state[:, 7:13] = 0.0
    ball.write_root_state_to_sim(ball_state)
    env.sim.forward()
    env.step(torch.zeros((1, ACTION_DIM), device=env.device))
    return {
        "real_defender_legal_contact_found": defender_contact,
        "attacker_to_defender_transfer_recorded": turnover,
        "post_tackle_control_timer_started": control_started,
        "clearance_after_transfer_m": float(env._v048_max_clearance.max()),
        "clearance_gate_reached": bool((env._v048_max_clearance >= MINIMUM_CLEARANCE_M).all()),
    }


def main() -> int:
    from mjlab.envs import ManagerBasedRlEnv

    cfg = exact_rcss_tackle_env_cfg()
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        observations, _ = env.reset(seed=48000)
        attacker = env.scene["attacker"]
        defender = env.scene["defender"]
        initial_attacker_touch = bool(
            (env.scene["attacker_ball_legal"].data.found > 0).any()
        )
        initial_defender_touch = bool(
            (env.scene["defender_ball_legal"].data.found > 0).any()
        )
        env.step(torch.zeros((1, ACTION_DIM), device=env.device))
        tracker_records_attacker_first = bool(
            env._v048_attacker_first_touch.all()
            and (env._v048_first_touch_owner == 1).all()
            and (env._v048_last_toucher == 1).all()
            and not env._v048_defender_touched.any()
            and not env._v048_turnover.any()
        )
        transfer = _real_transfer_probe(env)
        ball_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball/ball")
        pitch_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "pitch")
        action_term = env.action_manager.get_term("defender_fullbody")
        action_fixtures = np.stack(
            (
                np.full(ACTION_DIM, -2.0, dtype=np.float32),
                np.zeros(ACTION_DIM, dtype=np.float32),
                np.full(ACTION_DIM, 2.0, dtype=np.float32),
            )
        )
        expected_targets = np.stack(
            [
                clamp_t1_targets(
                    np.clip(row, -1.0, 1.0) * 0.30
                    + env.scene["defender"].data.default_joint_pos[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                for row in action_fixtures
            ]
        )
        # The runtime target is the intersection of the canonical competition
        # limits and the installed XML's slightly rounded joint ranges.
        runtime_lower = action_term._lower[0].detach().cpu().numpy()
        runtime_upper = action_term._upper[0].detach().cpu().numpy()
        expected_targets = np.maximum(
            runtime_lower, np.minimum(runtime_upper, expected_targets)
        )
        torch_targets = []
        for row in action_fixtures:
            action_term.process_actions(
                torch.as_tensor(row, device=env.device).view(1, ACTION_DIM)
            )
            torch_targets.append(
                action_term.clamped_target[0].detach().cpu().numpy().copy()
            )
        finite_and_shape_rejection = {}
        for label, invalid in (
            (
                "nonfinite",
                torch.full((1, ACTION_DIM), float("nan"), device=env.device),
            ),
            ("wrong_shape", torch.zeros((1, ACTION_DIM - 1), device=env.device)),
        ):
            try:
                action_term.process_actions(invalid)
            except (FloatingPointError, ValueError):
                finite_and_shape_rejection[label] = True
            else:
                finite_and_shape_rejection[label] = False
        action_term.process_actions(torch.zeros((1, ACTION_DIM), device=env.device))
        actor_components = defender_local_observation_components(env)
        live_actor_layout = []
        offset = 0
        for name, value in actor_components.items():
            width = int(value.shape[-1])
            live_actor_layout.append((name, (offset, offset + width)))
            offset += width
        contact_history_shapes = {
            name: list(env.scene[name].data.force_history.shape)
            for name in (
                "attacker_ball_legal",
                "defender_ball_legal",
                "cross_robot_impact",
                "defender_upper_vs_attacker",
                "attacker_upper_vs_defender",
                "defender_lower_vs_attacker",
                "attacker_lower_vs_defender",
            )
        }
        model_joint_ranges = np.asarray(
            [
                env.sim.mj_model.jnt_range[
                    mujoco.mj_name2id(
                        env.sim.mj_model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        f"defender/{name}",
                    )
                ]
                for name in RCSS_JOINT_NAMES
            ]
        )
        model_joint_effort_limits = np.asarray(
            [
                max(
                    abs(value)
                    for value in env.sim.mj_model.jnt_actfrcrange[
                        mujoco.mj_name2id(
                            env.sim.mj_model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            f"defender/{name}",
                        )
                    ]
                )
                for name in RCSS_JOINT_NAMES
            ]
        )
        collision = _collision_probe(env)
        contract_path = (
            EXPERIMENT_ROOT
            / "configs"
            / "v048_competitive_acceptance_v2.json"
        ).resolve()
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        fixed = contract["fixed_validation"]
        heldout = contract["held_out_test"]
        fixed_seeds = set(
            range(fixed["seed_start_inclusive"], fixed["seed_end_exclusive"])
        )
        heldout_seeds = set(
            range(
                heldout["seed_start_inclusive"], heldout["seed_end_exclusive"]
            )
        )
        sources = runtime_control_sources(contract_path)
        evaluator_text = (
            EXPERIMENT_ROOT / "scripts" / "evaluate_v048.py"
        ).read_text(encoding="utf-8")
        warmstart_text = (EXPERIMENT_ROOT / "warmstart.py").read_text(
            encoding="utf-8"
        )
        checks = {
            "two_exact_rcss_t1_entities": set(("attacker", "defender")).issubset(env.scene.entities),
            "each_t1_has_23_hinge_joints": len(attacker.joint_names) == 23 and len(defender.joint_names) == 23,
            "both_joint_orders_match_rcss": tuple(attacker.joint_names) == RCSS_JOINT_NAMES and tuple(defender.joint_names) == RCSS_JOINT_NAMES,
            "defender_policy_has_exactly_23_actions": env.action_manager.total_action_dim == ACTION_DIM,
            "defender_action_order_is_rcss_order": tuple(action_term.target_names) == RCSS_JOINT_NAMES,
            "defender_uses_fail_closed_safe_action_term": type(action_term).__name__
            == "V048SafeT1JointPositionAction",
            "safe_action_matches_canonical_clamp_t1_targets": np.allclose(
                np.stack(torch_targets), expected_targets, atol=1.0e-7
            ),
            "safe_action_rejects_nonfinite_and_wrong_shape": all(
                finite_and_shape_rejection.values()
            ),
            "actor_observation_is_93_local_features": observations["actor"].shape[-1] == ACTOR_OBSERVATION_DIM,
            "critic_has_strictly_more_information": observations["critic"].shape[-1] > observations["actor"].shape[-1],
            "live_actor_component_layout_matches_executable_contract": tuple(
                live_actor_layout
            )
            == V048_ACTOR_OBSERVATION_LAYOUT,
            "ball_geom_exists": ball_id >= 0,
            "ball_radius_is_exact": bool(np.isclose(env.sim.mj_model.geom_size[ball_id, 0], 0.11, atol=1e-12)),
            "ball_mass_is_exact": bool(np.isclose(env.sim.mj_model.body_mass[env.sim.mj_model.geom_bodyid[ball_id]], 0.41, atol=1e-12)),
            "ball_friction_is_exact": bool(np.allclose(env.sim.mj_model.geom_friction[ball_id], [0.4, 0.01, 0.01], atol=1e-12)),
            "ball_contact_semantics_are_exact": bool(
                env.sim.mj_model.geom_condim[ball_id] == 6
                and env.sim.mj_model.geom_priority[ball_id] == 1
                and np.allclose(
                    env.sim.mj_model.geom_solref[ball_id],
                    [-5000.0, -20.0],
                    atol=1e-12,
                )
            ),
            "pitch_size_and_timestep_are_exact": bool(
                pitch_id >= 0
                and np.allclose(
                    env.sim.mj_model.geom_size[pitch_id],
                    [32.0, 24.0, 40.0],
                    atol=1e-12,
                )
                and np.isclose(env.physics_dt, 0.005, atol=1e-12)
            ),
            "installed_xml_joint_limits_retained": bool(
                np.max(np.abs(model_joint_ranges[:, 0] - T1_LOWER_LIMITS)) <= 0.007
                and np.max(np.abs(model_joint_ranges[:, 1] - T1_UPPER_LIMITS)) <= 0.007
            ),
            "torque_safety_uses_exact_rcss_mechanical_caps": bool(
                np.allclose(model_joint_effort_limits, EFFORT_LIMITS.cpu().numpy(), atol=1e-12)
            ),
            "reset_starts_with_real_attacker_legal_ball_contact": initial_attacker_touch,
            "reset_does_not_fabricate_defender_ball_contact": not initial_defender_touch,
            "tracker_attributes_real_attacker_first_touch": tracker_records_attacker_first,
            "real_defender_legal_contact_detected": bool(transfer["real_defender_legal_contact_found"]),
            "real_contact_changes_ownership": bool(transfer["attacker_to_defender_transfer_recorded"]),
            "real_transfer_starts_control_timer": bool(transfer["post_tackle_control_timer_started"]),
            "clearance_is_measured_only_after_transfer": bool(transfer["clearance_gate_reached"]),
            "real_cross_robot_contact_detected": bool(collision["cross_robot_contact_found"]),
            "real_cross_robot_force_positive": float(collision["cross_robot_max_force_n"]) > 0.0,
            "head_torso_contact_reaches_foul_sensor": bool(collision["upper_body_contact_found"]),
            "lower_body_contact_reaches_foul_sensor": bool(collision["lower_body_contact_found"]),
            "impact_probe_remains_finite": bool(collision["finite_after_impact"]),
            "minimum_clearance_is_0_25m": MINIMUM_CLEARANCE_M == 0.25,
            "post_tackle_control_is_2s": MINIMUM_CONTROL_SECONDS == 2.0,
            "all_contact_sensors_store_four_physics_substeps": all(
                shape[2] == 4 for shape in contact_history_shapes.values()
            ),
            "contact_history_matches_policy_decimation": env.cfg.decimation == 4
            and all(shape[2] == env.cfg.decimation for shape in contact_history_shapes.values()),
            "actual_torque_metric_runs_each_physics_substep": (
                "maximum_actual_torque_ratio_per_physics_substep"
                in env.metrics_manager.active_terms
                and env.metrics_manager.cfg[
                    "maximum_actual_torque_ratio_per_physics_substep"
                ].per_substep
            ),
            "both_robots_accumulate_all_23_actual_joint_torques": (
                env._v048_attacker_max_abs_torque.shape == (1, 23)
                and env._v048_defender_max_abs_torque.shape == (1, 23)
                and bool(torch.isfinite(env._v048_attacker_max_abs_torque).all())
                and bool(torch.isfinite(env._v048_defender_max_abs_torque).all())
            ),
            "v1_contract_preserved": (
                EXPERIMENT_ROOT / "configs" / "v048_competitive_acceptance.json"
            ).is_file(),
            "v2_contract_has_8_by_5_grid": (
                contract.get("schema_version") == "v048_competitive_acceptance_v2"
                and contract["scenario_grid"]["direction_bins"] == 8
                and contract["scenario_grid"]["speed_bins"] == 5
                and contract["scenario_grid"]["cells"] == 40
            ),
            "v2_fixed_has_three_samples_per_cell": (
                len(fixed_seeds) == 120
                and fixed["samples_per_direction_speed_cell"] == 3
            ),
            "v2_heldout_has_ten_samples_per_cell": (
                len(heldout_seeds) == 400
                and heldout["samples_per_direction_speed_cell"] == 10
            ),
            "fixed_and_heldout_seeds_are_disjoint": fixed_seeds.isdisjoint(
                heldout_seeds
            ),
            "evaluator_binds_checkpoint_contract_sources_audit_and_fixed_proof": all(
                token in evaluator_text
                for token in (
                    "_valid_fixed_proof",
                    "runtime_control_sources",
                    "attacker_then_defender_transfer",
                    "maximum_continuous_control_seconds",
                    "attacker_max_abs_actual_torque_nm_by_joint",
                    "same_checkpoint_passing_fixed_proof",
                    "all_contact_histories_cover_four_substeps",
                )
            ),
            "warmstart_uses_semantic_reordering_not_first_75_copy": all(
                token in warmstart_text
                for token in (
                    "ACTOR_MAPPING",
                    '("base_ang_vel", (0, 3), (3, 6))',
                    '("projected_gravity", (3, 6), (0, 3))',
                    '("base_lin_vel", (75, 78), (6, 9))',
                    "formal_source_binding",
                )
            ),
        }
        passed = bool(checks) and all(checks.values())
        payload = {
            "schema_version": "v048_semantic_audit_v2",
            "version": "v048",
            "kind": "exact_rcss_two_t1_semantic_and_collision_audit",
            "task_source": str(EXPERIMENT_ROOT / "rcss_mjlab_task.py"),
            "task_source_sha256": _sha256(EXPERIMENT_ROOT / "rcss_mjlab_task.py"),
            "rcss_t1_xml": str(RCSS_T1_XML),
            "rcss_t1_xml_sha256": _sha256(RCSS_T1_XML),
            "actor_observation_dim": int(observations["actor"].shape[-1]),
            "critic_observation_dim": int(observations["critic"].shape[-1]),
            "action_dim": int(env.action_manager.total_action_dim),
            "effective_joint_effort_limits_nm": EFFORT_LIMITS.tolist(),
            "sources": sources,
            "acceptance_contract": {
                "path": str(contract_path),
                "sha256": _sha256(contract_path),
                "fixed_episodes": len(fixed_seeds),
                "heldout_episodes": len(heldout_seeds),
            },
            "live_actor_observation_layout": [
                [name, list(bounds)] for name, bounds in live_actor_layout
            ],
            "contact_force_history_shapes": contact_history_shapes,
            "collision_probe": collision,
            "transfer_probe": transfer,
            "checks": checks,
            "semantic_audit_passed": passed,
            "quality_passed": False,
            "conclusion": "Infrastructure semantics only; no tackle checkpoint has been trained or promoted."
        }
    finally:
        env.close()
    path = write_new_json("v048_semantic_audit", payload)
    print(json.dumps({"evidence": str(path), **payload}, ensure_ascii=False, indent=2))
    return 0 if payload["semantic_audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
