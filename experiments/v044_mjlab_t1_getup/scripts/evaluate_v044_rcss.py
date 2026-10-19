"""Strict four-pose RCSS evaluation for a v044 getup checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v044_mjlab_t1_getup.mjlab_getup_policy import (  # noqa: E402
    HOME_QPOS,
    sha256_file,
)
from experiments.v044_mjlab_t1_getup.rcss_getup_controller import (  # noqa: E402
    RcssMjlabGetupController,
)
from locomotion.official_t1_policy import MotionCommand  # noqa: E402
from locomotion.rcss_simulation import make_single_t1_simulation, torso_upright  # noqa: E402
from locomotion.rcss_t1_controller import RcssT1Controller, read_joint_state  # noqa: E402
from locomotion.t1_spec import (  # noqa: E402
    T1_LOWER_LIMITS,
    T1_UPPER_LIMITS,
    clamp_t1_targets,
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
UNSAFE_FORCE_N = 900.0


def _axis_angle(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    axis_array /= np.linalg.norm(axis_array)
    return np.concatenate(
        [[math.cos(angle / 2.0)], axis_array * math.sin(angle / 2.0)]
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


def _sha256_json(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _next_output(checkpoint: Path, split: str) -> Path:
    stem = checkpoint.stem
    for run in range(1, 1000):
        candidate = (
            EXPERIMENT_ROOT
            / "metrics"
            / f"v044_{stem}_{split}_run_{run:02d}.json"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError("no free metric run number")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("fixed_validation", "held_out_test"),
        default="fixed_validation",
    )
    parser.add_argument("--episodes-per-pose", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--fixed-proof", type=Path)
    parser.add_argument("--locomotion-model", type=Path, default=DEFAULT_LOCOMOTION_MODEL)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--recovery-timeout-s", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-walk", action="store_true")
    args = parser.parse_args()
    if args.episodes_per_pose is None:
        args.episodes_per_pose = 20 if args.split == "fixed_validation" else 100
    if args.seed_start is None:
        args.seed_start = 440100 if args.split == "fixed_validation" else 10440100
    if args.episodes_per_pose <= 0:
        parser.error("episodes-per-pose must be positive")
    if args.split == "held_out_test" and args.fixed_proof is None:
        parser.error("held_out_test requires --fixed-proof")
    return args


def _validate_fixed_proof(path: Path, checkpoint_sha256: str) -> dict[str, object]:
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("evaluation_split") != "fixed_validation":
        raise ValueError("fixed proof has the wrong split")
    if proof.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("fixed proof checkpoint hash mismatch")
    if not proof.get("validation_passed", False):
        raise ValueError("fixed proof did not pass")
    if int(proof.get("episodes_per_pose", 0)) < 20:
        raise ValueError("fixed proof has fewer than 20 episodes per pose")
    return {"path": str(path.resolve()), "sha256": _sha256_json(path)}


def _contact_snapshot(simulation: object, agent: object) -> dict[str, object]:
    model = simulation.mj_model
    data = simulation.mj_data
    prefix = agent.agent_id.prefix
    feet = {"left": False, "right": False}
    maximum = {"head": 0.0, "hand_or_torso": 0.0, "all": 0.0}
    wrench = np.zeros(6, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        body_names: list[str] = []
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            body_id = int(model.geom_bodyid[geom_id])
            body_names.append(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            )
        robot_names = [name for name in body_names if name.startswith(prefix)]
        if not robot_names:
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        force = float(np.linalg.norm(wrench[:3]))
        maximum["all"] = max(maximum["all"], force)
        for name in robot_names:
            lowered = name.lower()
            if "left_foot" in lowered:
                feet["left"] = True
            if "right_foot" in lowered:
                feet["right"] = True
            if "head" in lowered or name.endswith("H1") or name.endswith("H2"):
                maximum["head"] = max(maximum["head"], force)
            if (
                "hand" in lowered
                or "torso" in lowered
                or "waist" in lowered
            ):
                maximum["hand_or_torso"] = max(
                    maximum["hand_or_torso"], force
                )
    return {"feet": feet, "maximum_force_n": maximum}


def _robot_geom_ids(model: mujoco.MjModel, prefix: str) -> list[int]:
    result = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is not None and body_name.startswith(prefix):
            result.append(geom_id)
    return result


def _place_pose(
    simulation: object,
    agent: object,
    pose: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    axis, angle = POSES[pose]
    base = _axis_angle(axis, angle)
    yaw = _axis_angle((0.0, 0.0, 1.0), float(rng.uniform(-math.pi, math.pi)))
    jitter_x = _axis_angle((1.0, 0.0, 0.0), float(rng.uniform(-0.04, 0.04)))
    jitter_y = _axis_angle((0.0, 1.0, 0.0), float(rng.uniform(-0.04, 0.04)))
    quat = _quat_mul(yaw, _quat_mul(jitter_y, _quat_mul(jitter_x, base)))
    quat /= np.linalg.norm(quat)
    joints = clamp_t1_targets(
        HOME_QPOS + rng.uniform(-0.035, 0.035, size=23).astype(np.float32)
    ).astype(np.float64)
    agent.place_at((0.0, 0.0, 0.0), tuple(float(x) for x in quat))
    agent.init_joints(pos=joints)
    simulation.ball.drop_at(20.0, 0.0)
    simulation.mj_data.ctrl[:] = 0.0
    mujoco.mj_forward(simulation.mj_model, simulation.mj_data)

    pitch_id = mujoco.mj_name2id(
        simulation.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "pitch"
    )
    distances = []
    from_to = np.zeros(6, dtype=np.float64)
    for geom_id in _robot_geom_ids(simulation.mj_model, agent.agent_id.prefix):
        distances.append(
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
        )
    minimum_at_zero = min(distances)
    root_height = -minimum_at_zero + 0.005
    agent._qpos[2] = root_height
    agent._qvel[:] = 0.0
    simulation.mj_data.ctrl[:] = 0.0
    mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
    joint_pos, _ = read_joint_state(agent)
    within_limits = bool(
        np.all(joint_pos >= T1_LOWER_LIMITS - 1e-4)
        and np.all(joint_pos <= T1_UPPER_LIMITS + 1e-4)
    )
    return {
        "root_height_m": float(root_height),
        "minimum_geom_clearance_m": 0.005,
        "joint_limits_valid": within_limits,
        "finite_state": bool(
            np.isfinite(simulation.mj_data.qpos).all()
            and np.isfinite(simulation.mj_data.qvel).all()
        ),
    }


def _episode(
    *,
    checkpoint: Path,
    locomotion_model: Path,
    pose: str,
    seed: int,
    settle_s: float,
    recovery_timeout_s: float,
    skip_walk: bool,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    simulation, agent = make_single_t1_simulation()
    getup = RcssMjlabGetupController(checkpoint)
    locomotion = RcssT1Controller(locomotion_model)
    initial = _place_pose(simulation, agent, pose, rng)
    joint_pos, _ = read_joint_state(agent)
    getup.reset(joint_pos)
    dt = float(simulation.timestep)
    substeps = max(1, round(POLICY_DT / dt))

    # Match the official one-second settle window, but start only 5 mm above ground.
    settle_joint_violation = False
    for policy_index in range(max(1, round(settle_s / POLICY_DT))):
        del policy_index
        getup.hold_current(agent)
        for _ in range(substeps):
            getup.physics_step(simulation, agent)
            simulation.step([])
            qpos, _ = read_joint_state(agent)
            settle_joint_violation |= bool(
                np.any(qpos < T1_LOWER_LIMITS - 1e-4)
                or np.any(qpos > T1_UPPER_LIMITS + 1e-4)
            )

    maximum_force = {"head": 0.0, "hand_or_torso": 0.0, "all": 0.0}
    joint_violation = False
    nonfinite_action = False
    saturation_sum = 0.0
    physics_steps = 0
    continuous_stand_s = 0.0
    recovered = False
    recovery_elapsed_s = 0.0
    maximum_height = float(agent.xpos[2])
    maximum_upright = float(torso_upright(agent))
    for _ in range(max(1, round(recovery_timeout_s / POLICY_DT))):
        try:
            getup.policy_step(simulation, agent)
        except FloatingPointError:
            nonfinite_action = True
            break
        for _ in range(substeps):
            torque = getup.physics_step(simulation, agent)
            saturation_sum += float(torque["saturation_fraction"])
            physics_steps += 1
            simulation.step([])
            recovery_elapsed_s += dt
            contact = _contact_snapshot(simulation, agent)
            for key in maximum_force:
                maximum_force[key] = max(
                    maximum_force[key],
                    float(contact["maximum_force_n"][key]),
                )
            joint_pos, _ = read_joint_state(agent)
            joint_violation |= bool(
                np.any(joint_pos < T1_LOWER_LIMITS - 1e-4)
                or np.any(joint_pos > T1_UPPER_LIMITS + 1e-4)
            )
            height = float(agent.xpos[2])
            upright = float(torso_upright(agent))
            speed = float(np.linalg.norm(agent._qvel[:3]))
            maximum_height = max(maximum_height, height)
            maximum_upright = max(maximum_upright, upright)
            standing = bool(
                height >= 0.60
                and upright >= 0.90
                and speed <= 0.45
                and contact["feet"]["left"]
                and contact["feet"]["right"]
            )
            continuous_stand_s = continuous_stand_s + dt if standing else 0.0
            if continuous_stand_s >= 3.0 - 0.5 * dt:
                recovered = True
                break
        if recovered:
            break

    walk_success = False
    walk_distance_m = 0.0
    walk_fell = False
    if recovered and not skip_walk:
        start_xy = np.asarray(agent.xpos[:2], dtype=np.float64).copy()
        joint_pos, _ = read_joint_state(agent)
        locomotion.policy.reset()
        locomotion.filtered_targets[:] = joint_pos
        policy_time = 0.0
        command = MotionCommand(
            vx=0.25, vy=0.0, yaw_rate=0.0, head_yaw=0.0, head_pitch=0.28
        )
        for _ in range(round(5.0 / dt)):
            locomotion.command_step(simulation, agent, command, policy_time)
            simulation.step([])
            policy_time += dt
            if float(agent.xpos[2]) < 0.34 or float(torso_upright(agent)) < 0.20:
                walk_fell = True
        walk_distance_m = float(
            np.linalg.norm(np.asarray(agent.xpos[:2], dtype=np.float64) - start_xy)
        )
        walk_success = bool(not walk_fell and walk_distance_m >= 1.0)

    unsafe_head = maximum_force["head"] > UNSAFE_FORCE_N
    unsafe_body = maximum_force["hand_or_torso"] > UNSAFE_FORCE_N
    return {
        "pose": pose,
        "seed": seed,
        "initial_pose_contract": initial,
        "settle_joint_limit_violation": settle_joint_violation,
        "recovered": recovered,
        "recovery_elapsed_s": recovery_elapsed_s,
        "continuous_stand_s": continuous_stand_s,
        "maximum_height_m": maximum_height,
        "maximum_upright": maximum_upright,
        "walk_evaluated": bool(recovered and not skip_walk),
        "walk_success": walk_success,
        "walk_distance_m": walk_distance_m,
        "walk_fell": walk_fell,
        "joint_limit_violation": joint_violation,
        "nonfinite_action": nonfinite_action,
        "unsafe_head_contact": unsafe_head,
        "unsafe_hand_or_torso_contact": unsafe_body,
        "maximum_contact_force_n": maximum_force,
        "torque_saturation_fraction": (
            saturation_sum / physics_steps if physics_steps else 1.0
        ),
        "success": bool(
            recovered
            and not skip_walk
            and walk_success
            and not settle_joint_violation
            and not joint_violation
            and not nonfinite_action
            and not unsafe_head
            and not unsafe_body
        ),
    }


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    locomotion_model = args.locomotion_model.resolve()
    if not checkpoint.is_file() or not locomotion_model.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_file() else locomotion_model)
    checkpoint_sha = sha256_file(checkpoint)
    fixed_proof = None
    if args.split == "held_out_test":
        fixed_proof = _validate_fixed_proof(args.fixed_proof.resolve(), checkpoint_sha)
    output = args.output.resolve() if args.output else _next_output(checkpoint, args.split)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical evidence: {output}")

    episodes: list[dict[str, object]] = []
    for pose_index, pose in enumerate(POSES):
        for episode_index in range(args.episodes_per_pose):
            seed = args.seed_start + pose_index * args.episodes_per_pose + episode_index
            episode = _episode(
                checkpoint=checkpoint,
                locomotion_model=locomotion_model,
                pose=pose,
                seed=seed,
                settle_s=args.settle_s,
                recovery_timeout_s=args.recovery_timeout_s,
                skip_walk=args.skip_walk,
            )
            episodes.append(episode)
            print(json.dumps({
                "pose": pose,
                "seed": seed,
                "recovered": episode["recovered"],
                "walk_success": episode["walk_success"],
                "success": episode["success"],
            }, ensure_ascii=False), flush=True)

    count = len(episodes)
    per_pose = {}
    for pose in POSES:
        selected = [episode for episode in episodes if episode["pose"] == pose]
        per_pose[pose] = {
            "episodes": len(selected),
            "recovery_rate": float(np.mean([e["recovered"] for e in selected])),
            "success_rate": float(np.mean([e["success"] for e in selected])),
        }
    aggregate = {
        "episodes": count,
        "recovery_rate": float(np.mean([e["recovered"] for e in episodes])),
        "post_recovery_walk_rate": float(
            np.mean([e["walk_success"] for e in episodes])
        ),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "fall_rate": float(np.mean([e["walk_fell"] for e in episodes])),
        "joint_limit_violation_rate": float(
            np.mean([e["joint_limit_violation"] for e in episodes])
        ),
        "settle_joint_limit_violation_rate": float(
            np.mean([e["settle_joint_limit_violation"] for e in episodes])
        ),
        "nonfinite_action_rate": float(
            np.mean([e["nonfinite_action"] for e in episodes])
        ),
        "unsafe_head_contact_rate": float(
            np.mean([e["unsafe_head_contact"] for e in episodes])
        ),
        "unsafe_hand_or_torso_contact_rate": float(
            np.mean([e["unsafe_hand_or_torso_contact"] for e in episodes])
        ),
        "mean_torque_saturation_fraction": float(
            np.mean([e["torque_saturation_fraction"] for e in episodes])
        ),
    }
    gates = {
        "required_episode_count": args.episodes_per_pose >= (
            20 if args.split == "fixed_validation" else 100
        ),
        "all_initial_poses_valid": all(
            e["initial_pose_contract"]["joint_limits_valid"]
            and e["initial_pose_contract"]["finite_state"]
            for e in episodes
        ),
        "overall_recovery_rate": aggregate["recovery_rate"] >= 0.97,
        "per_pose_recovery_rate": all(
            metrics["recovery_rate"] >= 0.95 for metrics in per_pose.values()
        ),
        "post_recovery_walk_rate": aggregate["post_recovery_walk_rate"] >= 0.95,
        "fall_rate": aggregate["fall_rate"] <= 0.0,
        "joint_limit_violation_rate": (
            aggregate["joint_limit_violation_rate"] <= 0.0
            and aggregate["settle_joint_limit_violation_rate"] <= 0.0
        ),
        "nonfinite_action_rate": aggregate["nonfinite_action_rate"] <= 0.0,
        "unsafe_head_contact_rate": aggregate["unsafe_head_contact_rate"] <= 0.0,
        "unsafe_hand_or_torso_contact_rate": (
            aggregate["unsafe_hand_or_torso_contact_rate"] <= 0.01
        ),
        "torque_saturation_fraction": (
            aggregate["mean_torque_saturation_fraction"] <= 0.02
        ),
        "walk_was_not_skipped": not args.skip_walk,
    }
    stage_passed = all(gates.values())
    result = {
        "schema_version": "v044_rcss_getup_eval_v1",
        "evaluation_split": args.split,
        "seed_start": args.seed_start,
        "episodes_per_pose": args.episodes_per_pose,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
        },
        "locomotion_model": {
            "path": str(locomotion_model),
            "sha256": sha256_file(locomotion_model),
        },
        "fixed_validation_proof": fixed_proof,
        "settle_s": args.settle_s,
        "recovery_timeout_s": args.recovery_timeout_s,
        "episodes": episodes,
        "per_pose": per_pose,
        "aggregate": aggregate,
        "gates": gates,
        "validation_passed": bool(
            args.split == "fixed_validation" and stage_passed
        ),
        "quality_passed": bool(args.split == "held_out_test" and stage_passed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "aggregate": aggregate,
        "validation_passed": result["validation_passed"],
        "quality_passed": result["quality_passed"],
    }, ensure_ascii=False))
    return 0 if stage_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
