"""Audit v039 joint-limit evidence and standalone/RCSS model semantics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v039_getup_recovery.artifacts import write_json  # noqa: E402
from experiments.v039_getup_recovery.getup_env import (  # noqa: E402
    STATE_JOINT_LIMIT_TOLERANCE_RAD,
    T1GetUpEnv,
    _fullbody_model_path,
    joint_limit_overshoot,
)
from locomotion.rcss_simulation import make_single_t1_simulation  # noqa: E402
from locomotion.t1_spec import (  # noqa: E402
    T1_JOINT_NAMES,
    T1_LOWER_LIMITS,
    T1_UPPER_LIMITS,
)


def _named_violations(qpos: np.ndarray) -> list[dict[str, float | str]]:
    overshoot = joint_limit_overshoot(qpos)
    return [
        {
            "joint": T1_JOINT_NAMES[index],
            "qpos_rad": float(qpos[index]),
            "overshoot_rad": float(overshoot[index]),
        }
        for index in np.flatnonzero(overshoot > STATE_JOINT_LIMIT_TOLERANCE_RAD)
    ]


def _serial_ranges(env: T1GetUpEnv) -> np.ndarray:
    ranges = []
    for name in T1_JOINT_NAMES:
        runtime_name = "AAHead_yaw" if name == "Head_yaw" else name
        joint_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, runtime_name
        )
        ranges.append(env.model.jnt_range[joint_id].copy())
    return np.asarray(ranges, dtype=np.float64)


def _pose_probe(pose: str, seed: int) -> dict[str, object]:
    env = T1GetUpEnv(stage=3, fixed_pose=pose)
    try:
        _, reset_info = env.reset(seed=seed)
        reset_qpos, _ = env._joint_state()
        reset_root_z = float(env.data.qpos[2])
        reset_violations = _named_violations(reset_qpos.copy())
        first_violation_by_joint: dict[str, dict[str, float | int]] = {}
        final_info: dict[str, object] = {}
        for step in range(20):
            _, _, terminated, truncated, final_info = env.step(
                np.zeros(23, dtype=np.float32)
            )
            qpos, _ = env._joint_state()
            for item in _named_violations(qpos.copy()):
                joint = str(item["joint"])
                first_violation_by_joint.setdefault(
                    joint,
                    {
                        "control_step": step + 1,
                        "qpos_rad": float(item["qpos_rad"]),
                        "overshoot_rad": float(item["overshoot_rad"]),
                    },
                )
            if terminated or truncated:
                break
        return {
            "pose": pose,
            "seed": seed,
            "configured_root_z_before_settle_m": 0.44,
            "root_z_after_unactuated_0p30s_settle_m": reset_root_z,
            "root_z_drop_during_settle_m": float(0.44 - reset_root_z),
            "reset_info": reset_info,
            "reset_state_violations": reset_violations,
            "first_20_step_violations": first_violation_by_joint,
            "episode_joint_limit_violation": bool(
                final_info.get("joint_limit_violation", False)
            ),
            "max_state_joint_overshoot_rad": float(
                final_info.get("max_joint_overshoot_rad", 0.0)
            ),
            "requested_target_was_clamped": bool(
                final_info.get("requested_target_was_clamped", False)
            ),
            "sent_target_limit_violation": bool(
                final_info.get("sent_target_limit_violation", True)
            ),
            "max_sent_target_overshoot_rad": float(
                final_info.get("max_sent_target_overshoot_rad", float("inf"))
            ),
            "all_23_targets": bool(final_info.get("all_23_targets", False)),
        }
    finally:
        env.close()


def _target_clamp_probe() -> dict[str, object]:
    """Send both extreme signs on all 23 channels and inspect the actual target."""

    env = T1GetUpEnv(stage=0)
    try:
        env.reset(seed=39_900)
        samples = []
        for sign in (1.0, -1.0):
            env._apply_action(np.full(23, sign * 10.0, dtype=np.float32))
            overshoot = joint_limit_overshoot(env.filtered_target)
            samples.append(
                {
                    "input_value_before_action_space_clip": sign * 10.0,
                    "requested_target_was_clamped": env.requested_target_was_clamped,
                    "all_23_targets": bool(env.target_was_sent.all()),
                    "max_sent_target_overshoot_rad": float(np.max(overshoot)),
                }
            )
        return {
            "samples": samples,
            "passed": bool(
                env.target_was_sent.all()
                and env.requested_target_was_clamped
                and not env.sent_target_limit_violation
                and env.max_sent_target_overshoot_rad <= 1.0e-6
            ),
        }
    finally:
        env.close()


def main() -> None:
    serial_env = T1GetUpEnv(stage=0)
    try:
        serial_ranges = _serial_ranges(serial_env)
        serial_model = {
            "source": str(_fullbody_model_path()),
            "nq": int(serial_env.model.nq),
            "nv": int(serial_env.model.nv),
            "nu": int(serial_env.model.nu),
            "timestep_s": float(serial_env.model.opt.timestep),
            "body_mass_sum_kg": float(np.sum(serial_env.model.body_mass)),
            "actuator_semantics": "23 direct torque motors; v039 computes and clips PD torque before data.ctrl",
            "joint_ranges_rad": serial_ranges.tolist(),
        }
    finally:
        serial_env.close()

    simulation, agent = make_single_t1_simulation()
    rcss_model = simulation.mj_model
    rcss_ranges = np.asarray(
        [rcss_model.jnt_range[int(joint.id)].copy() for joint in agent._joints],
        dtype=np.float64,
    )
    prefix = agent.agent_id.prefix
    rcss_body_ids = [
        body_id
        for body_id in range(rcss_model.nbody)
        if (mujoco.mj_id2name(rcss_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").startswith(prefix)
    ]
    rcss_summary = {
        "nq_whole_match": int(rcss_model.nq),
        "nv_whole_match": int(rcss_model.nv),
        "nu_whole_match": int(rcss_model.nu),
        "timestep_s": float(rcss_model.opt.timestep),
        "robot_body_mass_sum_kg": float(np.sum(rcss_model.body_mass[rcss_body_ids])),
        "robot_joint_count": len(agent._joints),
        "robot_actuator_count": 3 * len(agent._joints),
        "actuator_semantics": (
            "three actuators per joint (tau + position + velocity); ctrl_motor configures PD gains; "
            "the competition controller handles ankle parallel-mechanism indexes separately"
        ),
        "joint_ranges_rad": rcss_ranges.tolist(),
    }

    locomotion_path = _fullbody_model_path().with_name("T1_locomotion.xml")
    locomotion_model = mujoco.MjModel.from_xml_path(str(locomotion_path))
    locomotion_summary = {
        "source": str(locomotion_path),
        "nq": int(locomotion_model.nq),
        "nv": int(locomotion_model.nv),
        "nu": int(locomotion_model.nu),
        "body_mass_sum_kg": float(np.sum(locomotion_model.body_mass)),
        "actuator_semantics": "12 leg-only direct torque motors; head, arms and waist are fixed into the trunk",
        "left_hip_roll_range_rad": locomotion_model.joint("Left_Hip_Roll").range.tolist(),
        "right_hip_roll_range_rad": locomotion_model.joint("Right_Hip_Roll").range.tolist(),
    }

    rule_ranges = np.column_stack([T1_LOWER_LIMITS, T1_UPPER_LIMITS]).astype(
        np.float64
    )
    poses = [_pose_probe(pose, 39_910) for pose in ("left_side", "right_side", "prone", "supine")]
    target_clamp_probe = _target_clamp_probe()
    payload = {
        "experiment": "v039_getup_recovery",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "Is joint_limit_violation_rate=1.0 a target-clamp bug or physical state excursion?",
        "answer": (
            "It is a real physical qpos excursion in the MuJoCo soft joint constraints, not an unclamped "
            "23-D target. The old reset logic actually under-reported it by clearing violations after settle."
        ),
        "state_joint_limit_tolerance_rad": STATE_JOINT_LIMIT_TOLERANCE_RAD,
        "models": {
            "standalone_T1_serial": serial_model,
            "rcss_full_T1": rcss_summary,
            "official_policy_training_T1_locomotion": locomotion_summary,
        },
        "range_comparison": {
            "standalone_vs_rcss_exact": bool(np.array_equal(serial_ranges, rcss_ranges)),
            "maximum_standalone_vs_rcss_delta_rad": float(
                np.max(np.abs(serial_ranges - rcss_ranges))
            ),
            "maximum_rule_vs_standalone_delta_rad": float(
                np.max(np.abs(rule_ranges - serial_ranges))
            ),
        },
        "pose_probes": poses,
        "target_clamp_probe": target_clamp_probe,
        "all_probes_sent_23_targets": all(bool(item["all_23_targets"]) for item in poses),
        "any_sent_target_limit_violation": any(
            bool(item["sent_target_limit_violation"]) for item in poses
        ),
        "any_physical_state_joint_limit_violation": any(
            bool(item["episode_joint_limit_violation"]) for item in poses
        ),
        "initial_pose_contract_passed": all(
            not bool(item["reset_info"]["initial_joint_limit_violation"])
            for item in poses
        ),
        "handoff_compatibility_passed": False,
        "handoff_blockers": [
            "The official T1.pt was trained against the 12-actuator fixed-upper-body locomotion model.",
            "T1_serial and RCSS differ in mass, timestep, joint friction, torso/foot collision geoms, and actuator API.",
            "The official policy must be adapted or distilled and then validated in the complete RCSS T1 model.",
        ],
        "quality_passed": False,
    }
    output = write_json(
        Path(__file__).resolve().parents[1]
        / "metrics"
        / "v039_joint_limit_and_model_semantics_audit.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
