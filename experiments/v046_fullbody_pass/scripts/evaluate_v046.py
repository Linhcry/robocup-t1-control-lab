#!/usr/bin/env python3
"""Strict, replayable fixed/held-out evaluation for v046 passing policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_CONTRACT = (
    EXPERIMENT_ROOT / "configs" / "v046_competitive_acceptance_v2.json"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(EXPERIMENT_ROOT)): _sha256(path)
        for path in sorted(EXPERIMENT_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _next_output(checkpoint: Path, split: str, smoke: bool) -> Path:
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    label = "engineering_smoke" if smoke else split
    for run in range(1, 10_000):
        path = root / f"v046_{checkpoint.stem}_{label}_run_{run:02d}.json"
        if not path.exists():
            return path
    raise RuntimeError("no unused v046 evaluation filename")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("fixed_validation", "held_out_test"),
        default="fixed_validation",
    )
    parser.add_argument("--fixed-proof", type=Path)
    parser.add_argument("--acceptance-config", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error("episodes must be positive")
    if args.smoke and args.split != "fixed_validation":
        parser.error("engineering smoke is restricted to fixed_validation")
    if not args.smoke and (args.episodes is not None or args.seed_start is not None):
        parser.error("formal protocols forbid episode or seed overrides")
    if args.split == "held_out_test" and args.fixed_proof is None:
        parser.error("held_out_test requires --fixed-proof")
    return args


def _load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment",
        "task_id",
        "required_splits",
        "scenario_grid",
        "episode_contract",
        "quality_gates",
        "promotion",
    }
    if set(contract) != required:
        raise ValueError(f"acceptance contract keys differ: {sorted(set(contract) ^ required)}")
    if contract["schema_version"] != "v046_competitive_acceptance_v2":
        raise ValueError("acceptance contract schema mismatch")
    fixed = contract["required_splits"]["fixed_validation"]
    held = contract["required_splits"]["held_out_test"]
    fixed_seeds = set(range(fixed["seed_start"], fixed["seed_start"] + fixed["episodes"]))
    held_seeds = set(range(held["seed_start"], held["seed_start"] + held["episodes"]))
    if fixed_seeds & held_seeds:
        raise ValueError("fixed and held-out seed ranges overlap")
    return contract


def _all_true(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        item is True for item in value.values()
    )


def _validate_semantic_audit(
    path: Path, runtime_sources: dict[str, str]
) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != "v046_model_semantic_audit_v1":
        raise ValueError("semantic audit schema mismatch")
    if proof.get("model_semantics_passed") is not True or not _all_true(
        proof.get("checks")
    ):
        raise ValueError("semantic audit did not pass")
    if proof.get("source_sha256") != runtime_sources:
        raise ValueError("v046 source files changed after the semantic audit")
    return {"path": str(path), "sha256": _sha256(path)}


def _required_seeds(split_cfg: dict[str, Any]) -> list[int]:
    return list(
        range(
            int(split_cfg["seed_start"]),
            int(split_cfg["seed_start"]) + int(split_cfg["episodes"]),
        )
    )


def _validate_fixed_proof(
    path: Path,
    *,
    checkpoint_sha: str,
    contract_sha: str,
    semantic_audit_sha: str,
    runtime_sources: dict[str, str],
    expected_seeds: list[int],
    heldout_seeds: list[int],
) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    proof = json.loads(path.read_text(encoding="utf-8"))
    episodes = proof.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("fixed proof has no per-episode evidence")
    recorded_seeds = [int(row["seed"]) for row in episodes]
    checks = {
        "schema": proof.get("schema_version") == "v046_strict_pass_eval_v1",
        "split": proof.get("evaluation_split") == "fixed_validation",
        "formal": proof.get("formal_protocol") is True,
        "checkpoint": proof.get("checkpoint", {}).get("sha256") == checkpoint_sha,
        "contract": proof.get("acceptance_config", {}).get("sha256") == contract_sha,
        "audit": proof.get("semantic_audit", {}).get("sha256") == semantic_audit_sha,
        "sources": proof.get("source_sha256") == runtime_sources,
        "seeds": recorded_seeds == expected_seeds,
        "seed_hash": proof.get("episode_seed_sha256") == _json_hash(recorded_seeds),
        "gates": _all_true(proof.get("gate_checks")),
        "passed": proof.get("fixed_validation_passed") is True,
        "top_quality_remained_false": proof.get("quality_passed") is False,
        "complete_episode_rows": all(
            isinstance(row.get("contact_evidence"), dict)
            and isinstance(row.get("joint_evidence"), dict)
            and isinstance(row.get("actual_torque_evidence"), dict)
            and "nonfinite_detected" in row
            for row in episodes
        ),
        "heldout_disjoint": not set(recorded_seeds).intersection(heldout_seeds),
    }
    if not all(checks.values()):
        raise ValueError(f"fixed proof failed binding checks: {checks}")
    return {"path": str(path), "sha256": _sha256(path)}


def _scenario(
    index: int, seed: int, contract: dict[str, Any]
) -> dict[str, Any]:
    grid = contract["scenario_grid"]
    angle_edges = np.asarray(grid["target_angle_bin_edges_deg"], dtype=np.float64)
    distance_edges = np.asarray(grid["pass_distance_bin_edges_m"], dtype=np.float64)
    direction_bin = index % (len(angle_edges) - 1)
    distance_bin = (index // (len(angle_edges) - 1)) % (len(distance_edges) - 1)
    rng = np.random.default_rng(seed)
    angle_low, angle_high = angle_edges[direction_bin : direction_bin + 2]
    distance_low, distance_high = distance_edges[distance_bin : distance_bin + 2]
    # Stay away from exact bin edges so replay is independent of floating-point
    # digitize conventions.
    angle = float(rng.uniform(angle_low + 0.05, angle_high - 0.05))
    distance = float(rng.uniform(distance_low + 0.001, distance_high - 0.001))
    ball_x = float(rng.uniform(*grid["ball_x_range_m"]))
    ball_y = float(rng.uniform(*grid["ball_y_range_m"]))
    return {
        "seed": seed,
        "direction_bin": direction_bin,
        "distance_bin": distance_bin,
        "target_angle_deg": angle,
        "commanded_distance_m": distance,
        "ball_x_m": ball_x,
        "ball_y_m": ball_y,
    }


def _set_scenario(env: Any, scenario: dict[str, Any]) -> None:
    from experiments.v046_fullbody_pass.mdp.commands import DirectionalPassCommand

    command = env.command_manager.get_term("directional_pass")
    if not isinstance(command, DirectionalPassCommand):
        raise TypeError("directional_pass command has the wrong implementation")
    ball = env.scene["ball"]
    angle = math.radians(scenario["target_angle_deg"])
    direction = torch.tensor(
        [[math.cos(angle), math.sin(angle), 0.0]],
        device=env.device,
        dtype=torch.float32,
    )
    ball_pos = env.scene.env_origins.clone()
    ball_pos[0, 0] += scenario["ball_x_m"]
    ball_pos[0, 1] += scenario["ball_y_m"]
    ball_pos[0, 2] += command.cfg.ball_radius_m + command.cfg.ball_reset_clearance_m
    command.direction_w[:] = direction
    command.pass_distance[:] = scenario["commanded_distance_m"]
    command.initial_ball_pos_w[:] = ball_pos
    command.target_pos_w[:] = ball_pos + direction * scenario["commanded_distance_m"]
    command.foot_touched[:] = False
    command.upper_body_touched[:] = False
    command.episode_success[:] = False
    command.progress_m[:] = 0.0
    command.lateral_error_m[:] = 0.0
    pose = torch.zeros((1, 7), device=env.device)
    pose[:, :3] = ball_pos
    pose[:, 3] = 1.0
    ball.write_root_link_pose_to_sim(pose)
    ball.write_root_link_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    env.sim.forward()


def _sensor_channels(sensor: Any) -> tuple[np.ndarray, np.ndarray]:
    found = sensor.data.found
    force = sensor.data.force
    if found is None or force is None:
        raise RuntimeError(f"contact sensor {sensor.cfg.name} lacks found/force data")
    found_np = found[0].detach().cpu().numpy() > 0.0
    force_np = np.linalg.norm(force[0].detach().cpu().numpy(), axis=-1)
    force_history = sensor.data.force_history
    if force_history is None:
        raise RuntimeError(
            f"contact sensor {sensor.cfg.name} lacks physics-substep history"
        )
    history_np = np.linalg.norm(
        force_history[0].detach().cpu().numpy(), axis=-1
    )
    if history_np.ndim != 2 or history_np.shape[1] != 4:
        raise RuntimeError(
            f"contact sensor {sensor.cfg.name} history is not channel-by-4-substep"
        )
    found_np |= (history_np > 0.0).any(axis=-1)
    force_np = np.maximum(force_np, history_np.max(axis=-1))
    return found_np, force_np


def _direction_error_deg(displacement: np.ndarray, direction: np.ndarray) -> float:
    norm = float(np.linalg.norm(displacement[:2]))
    if norm <= 1.0e-9:
        return 180.0
    cosine = float(np.dot(displacement[:2] / norm, direction[:2]))
    return float(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))


def _episode(
    env: Any,
    wrapped: Any,
    policy: Any,
    scenario: dict[str, Any],
    horizon: int,
    episode_cfg: dict[str, Any],
) -> dict[str, Any]:
    from experiments.v046_fullbody_pass.mdp.commands import DirectionalPassCommand

    wrapped.seed(int(scenario["seed"]))
    wrapped.reset()
    _set_scenario(env, scenario)
    obs = wrapped.get_observations()
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    command = env.command_manager.get_term("directional_pass")
    if not isinstance(command, DirectionalPassCommand):
        raise TypeError("directional_pass command type mismatch")
    foot_sensor = env.scene["foot_ball_contact"]
    upper_sensor = env.scene["upper_body_ball_contact"]
    foot_names = list(foot_sensor.cfg.primary.pattern)
    upper_names = list(upper_sensor.cfg.primary.pattern)
    if len(foot_names) != 2 or len(upper_names) != 12:
        raise RuntimeError("contact sensor channel count differs from the contract")

    foot_seen = np.zeros(2, dtype=bool)
    upper_seen = np.zeros(12, dtype=bool)
    foot_peak_force = np.zeros(2, dtype=np.float64)
    upper_peak_force = np.zeros(12, dtype=np.float64)
    max_actual_torque = np.zeros(23, dtype=np.float64)
    torque_over_names: set[str] = set()
    torque_samples = 0
    saturated_samples = 0
    joint_violation_names: set[str] = set()
    max_joint_overshoot = 0.0
    fell = False
    nonfinite = False
    unexpected_done = False
    reward_finite = True
    maximum_progress = -float("inf")
    lateral_at_maximum_progress = float("inf")
    direction_error_at_maximum_progress = 180.0
    maximum_ball_height = 0.0
    success_step: int | None = None
    steps_executed = 0
    from locomotion.rcss_t1_controller import TORQUE_LIMIT
    from locomotion.t1_spec import T1_JOINT_NAMES

    direction = command.direction_w[0].detach().cpu().numpy().astype(np.float64)
    initial_ball = command.initial_ball_pos_w[0].detach().cpu().numpy().astype(np.float64)
    limits = np.asarray(TORQUE_LIMIT, dtype=np.float64)
    for step in range(horizon):
        actor_obs = obs["actor"]
        if not torch.isfinite(actor_obs).all():
            nonfinite = True
            break
        with torch.inference_mode():
            action = policy(obs)
        if tuple(action.shape) != (1, 23) or not torch.isfinite(action).all():
            nonfinite = True
            break
        obs, reward, done, _ = wrapped.step(action)
        steps_executed += 1
        reward_finite &= bool(torch.isfinite(reward).all().item())
        unexpected_done |= bool(done[0].item())

        foot_now, foot_force = _sensor_channels(foot_sensor)
        upper_now, upper_force = _sensor_channels(upper_sensor)
        foot_seen |= foot_now
        upper_seen |= upper_now
        foot_peak_force = np.maximum(foot_peak_force, foot_force)
        upper_peak_force = np.maximum(upper_peak_force, upper_force)

        joint_pos = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
        joint_vel = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float64)
        actual_torque = (
            robot.data.qfrc_actuator[0].detach().cpu().numpy().astype(np.float64)
        )
        root_pos = robot.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)
        gravity = (
            robot.data.projected_gravity_b[0].detach().cpu().numpy().astype(np.float64)
        )
        ball_pos = ball.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)
        ball_vel = ball.data.root_link_vel_w[0].detach().cpu().numpy().astype(np.float64)
        arrays = (joint_pos, joint_vel, actual_torque, root_pos, gravity, ball_pos, ball_vel)
        if not reward_finite or not all(np.isfinite(array).all() for array in arrays):
            nonfinite = True
            break

        max_actual_torque = np.maximum(max_actual_torque, np.abs(actual_torque))
        torque_samples += 23
        saturated_samples += int(
            np.sum(np.abs(actual_torque) >= limits * episode_cfg["torque_saturation_fraction"])
        )
        over = np.abs(actual_torque) > limits + episode_cfg["actual_torque_tolerance_nm"]
        torque_over_names.update(
            name for name, bad in zip(T1_JOINT_NAMES, over, strict=True) if bad
        )

        joint_limits = (
            robot.data.joint_pos_limits[0].detach().cpu().numpy().astype(np.float64)
        )
        overshoot = np.maximum(
            np.maximum(joint_limits[:, 0] - joint_pos, joint_pos - joint_limits[:, 1]),
            0.0,
        )
        max_joint_overshoot = max(max_joint_overshoot, float(np.max(overshoot)))
        joint_bad = overshoot > episode_cfg["joint_limit_tolerance_rad"]
        joint_violation_names.update(
            name for name, bad in zip(T1_JOINT_NAMES, joint_bad, strict=True) if bad
        )
        tilt = float(np.linalg.norm(gravity[:2]))
        fell |= bool(
            root_pos[2] < episode_cfg["fall_minimum_root_height_m"]
            or tilt > episode_cfg["fall_maximum_tilt_sine"]
        )

        displacement = ball_pos - initial_ball
        progress = float(np.dot(displacement, direction))
        lateral = float(
            np.linalg.norm(displacement[:2] - progress * direction[:2])
        )
        maximum_ball_height = max(maximum_ball_height, float(ball_pos[2]))
        if progress > maximum_progress:
            maximum_progress = progress
            lateral_at_maximum_progress = lateral
            direction_error_at_maximum_progress = _direction_error_deg(
                displacement, direction
            )
        if success_step is None and bool(command.success_now()[0].item()):
            success_step = step

    final_ball = ball.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)
    final_displacement = final_ball - initial_ball
    final_progress = float(np.dot(final_displacement, direction))
    final_lateral = float(
        np.linalg.norm(final_displacement[:2] - final_progress * direction[:2])
    )
    if maximum_progress == -float("inf"):
        maximum_progress = 0.0
        lateral_at_maximum_progress = 0.0
    commanded_distance = float(scenario["commanded_distance_m"])
    raw_success = success_step is not None
    upper_illegal = bool(upper_seen.any())
    joint_violation = bool(joint_violation_names)
    torque_violation = bool(torque_over_names)
    episode_complete = steps_executed == horizon and not unexpected_done and not nonfinite
    strict_success = bool(
        raw_success
        and foot_seen.any()
        and not upper_illegal
        and not fell
        and not joint_violation
        and not torque_violation
        and episode_complete
    )
    return {
        **scenario,
        "steps_executed": steps_executed,
        "required_steps": horizon,
        "episode_complete": episode_complete,
        "unexpected_done": unexpected_done,
        "reward_finite": reward_finite,
        "nonfinite_detected": nonfinite,
        "contact_evidence": {
            "real_foot_ball_contact": bool(foot_seen.any()),
            "foot_channels_touched": [
                name for name, touched in zip(foot_names, foot_seen, strict=True) if touched
            ],
            "peak_foot_force_n": {
                name: float(force)
                for name, force in zip(foot_names, foot_peak_force, strict=True)
            },
            "illegal_upper_body_ball_contact": upper_illegal,
            "upper_body_channels_touched": [
                name
                for name, touched in zip(upper_names, upper_seen, strict=True)
                if touched
            ],
            "peak_upper_body_force_n": {
                name: float(force)
                for name, force in zip(upper_names, upper_peak_force, strict=True)
            },
        },
        "pass_evidence": {
            "raw_contact_distance_corridor_success": raw_success,
            "strict_safe_success": strict_success,
            "success_step": success_step,
            "maximum_forward_progress_m": maximum_progress,
            "final_forward_progress_m": final_progress,
            "lateral_error_at_maximum_progress_m": lateral_at_maximum_progress,
            "final_lateral_error_m": final_lateral,
            "direction_error_at_maximum_progress_deg": direction_error_at_maximum_progress,
            "relative_distance_error": abs(maximum_progress - commanded_distance)
            / commanded_distance,
            "maximum_ball_height_m": maximum_ball_height,
        },
        "fall_evidence": {"fell": fell},
        "joint_evidence": {
            "violation": joint_violation,
            "violating_joint_names": sorted(joint_violation_names),
            "maximum_overshoot_rad": max_joint_overshoot,
        },
        "actual_torque_evidence": {
            "limits_nm": {
                name: float(limit)
                for name, limit in zip(T1_JOINT_NAMES, limits, strict=True)
            },
            "maximum_absolute_by_joint_nm": {
                name: float(value)
                for name, value in zip(T1_JOINT_NAMES, max_actual_torque, strict=True)
            },
            "limit_violation": torque_violation,
            "violating_joint_names": sorted(torque_over_names),
            "saturated_joint_sample_fraction": float(saturated_samples / max(1, torque_samples)),
            "joint_samples": torque_samples,
        },
    }


def _rate(rows: list[dict[str, Any]], selector) -> float:
    return float(np.mean([bool(selector(row)) for row in rows])) if rows else 0.0


def _bin_success(
    rows: list[dict[str, Any]], key: str, count: int
) -> tuple[dict[str, dict[str, float]], float]:
    result = {}
    rates = []
    for index in range(count):
        subset = [row for row in rows if row[key] == index]
        rate = _rate(subset, lambda row: row["pass_evidence"]["strict_safe_success"])
        result[str(index)] = {"episodes": len(subset), "success_rate": rate}
        rates.append(rate)
    return result, min(rates) if rates else 0.0


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    contract_path = args.acceptance_config.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = _load_contract(contract_path)
    runtime_sources = _runtime_source_hashes()
    semantic_audit = _validate_semantic_audit(args.semantic_audit, runtime_sources)
    split_cfg = contract["required_splits"][args.split]
    if args.smoke:
        episode_count = args.episodes if args.episodes is not None else 1
        seed_start = args.seed_start if args.seed_start is not None else 460001
        seeds = list(range(seed_start, seed_start + episode_count))
    else:
        seeds = _required_seeds(split_cfg)
    fixed_seeds = _required_seeds(contract["required_splits"]["fixed_validation"])
    fixed_proof = None
    if args.split == "held_out_test":
        fixed_proof = _validate_fixed_proof(
            args.fixed_proof,
            checkpoint_sha=_sha256(checkpoint),
            contract_sha=_sha256(contract_path),
            semantic_audit_sha=semantic_audit["sha256"],
            runtime_sources=runtime_sources,
            expected_seeds=fixed_seeds,
            heldout_seeds=seeds,
        )
    scenarios = [_scenario(i, seed, contract) for i, seed in enumerate(seeds)]

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from experiments.v046_fullbody_pass.rcss_mjlab_task import (
        rcss_t1_directional_pass_env_cfg,
        rcss_t1_directional_pass_ppo_runner_cfg,
    )

    env_cfg = rcss_t1_directional_pass_env_cfg(play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain.num_envs = 1
    env_cfg.seed = seeds[0]
    env_cfg.episode_length_s = contract["episode_contract"]["duration_s"] + 0.5
    # The evaluator records every safety condition itself.  Keeping only the
    # later time limit prevents automatic reset from erasing the violating
    # state or a true contact before the complete eight-second trace is saved.
    env_cfg.terminations = {"time_out": env_cfg.terminations["time_out"]}
    agent_cfg = rcss_t1_directional_pass_ppo_runner_cfg()
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    horizon = round(contract["episode_contract"]["duration_s"] / env.step_dt)
    rows = []
    try:
        for scenario in scenarios:
            rows.append(
                _episode(
                    env,
                    wrapped,
                    policy,
                    scenario,
                    horizon,
                    contract["episode_contract"],
                )
            )
    finally:
        wrapped.close()

    direction_bins, worst_direction = _bin_success(
        rows,
        "direction_bin",
        len(contract["scenario_grid"]["target_angle_bin_edges_deg"]) - 1,
    )
    distance_bins, worst_distance = _bin_success(
        rows,
        "distance_bin",
        len(contract["scenario_grid"]["pass_distance_bin_edges_m"]) - 1,
    )
    summary = {
        "episodes": len(rows),
        "true_foot_contact_rate": _rate(
            rows, lambda row: row["contact_evidence"]["real_foot_ball_contact"]
        ),
        "pass_success_rate": _rate(
            rows, lambda row: row["pass_evidence"]["strict_safe_success"]
        ),
        "raw_pass_success_rate": _rate(
            rows,
            lambda row: row["pass_evidence"]["raw_contact_distance_corridor_success"],
        ),
        "upper_body_ball_contact_rate": _rate(
            rows,
            lambda row: row["contact_evidence"]["illegal_upper_body_ball_contact"],
        ),
        "fall_rate": _rate(rows, lambda row: row["fall_evidence"]["fell"]),
        "joint_limit_violation_rate": _rate(
            rows, lambda row: row["joint_evidence"]["violation"]
        ),
        "actual_torque_limit_violation_rate": _rate(
            rows, lambda row: row["actual_torque_evidence"]["limit_violation"]
        ),
        "nonfinite_rate": _rate(rows, lambda row: row["nonfinite_detected"]),
        "incomplete_episode_rate": _rate(
            rows, lambda row: not row["episode_complete"]
        ),
        "mean_torque_saturation_fraction": float(
            np.mean(
                [
                    row["actual_torque_evidence"]["saturated_joint_sample_fraction"]
                    for row in rows
                ]
            )
        ),
        "median_direction_error_deg": float(
            np.median(
                [
                    row["pass_evidence"]["direction_error_at_maximum_progress_deg"]
                    for row in rows
                ]
            )
        ),
        "median_relative_distance_error": float(
            np.median(
                [row["pass_evidence"]["relative_distance_error"] for row in rows]
            )
        ),
        "worst_direction_bin_success_rate": worst_direction,
        "worst_distance_bin_success_rate": worst_distance,
    }
    gates = contract["quality_gates"]
    expected_count = split_cfg["episodes"] if not args.smoke else len(seeds)
    direction_minimum = contract["scenario_grid"][
        "minimum_heldout_samples_per_direction_bin"
        if args.split == "held_out_test"
        else "minimum_fixed_samples_per_direction_bin"
    ]
    distance_minimum = contract["scenario_grid"][
        "minimum_heldout_samples_per_distance_bin"
        if args.split == "held_out_test"
        else "minimum_fixed_samples_per_distance_bin"
    ]
    gate_checks = {
        "semantic_audit_matches_current_sources": True,
        "required_episode_count": len(rows) == expected_count,
        "all_direction_bins_present": all(
            item["episodes"] >= direction_minimum for item in direction_bins.values()
        ),
        "all_distance_bins_present": all(
            item["episodes"] >= distance_minimum for item in distance_bins.values()
        ),
        "true_foot_contact_rate": summary["true_foot_contact_rate"]
        >= gates["minimum_true_foot_contact_rate"],
        "strict_pass_success_rate": summary["pass_success_rate"]
        >= gates["minimum_pass_success_rate"],
        "upper_body_contact_rate": summary["upper_body_ball_contact_rate"]
        <= gates["maximum_upper_body_ball_contact_rate"],
        "fall_rate": summary["fall_rate"] <= gates["maximum_fall_rate"],
        "joint_limit_rate": summary["joint_limit_violation_rate"]
        <= gates["maximum_joint_limit_violation_rate"],
        "actual_torque_limit_rate": summary["actual_torque_limit_violation_rate"]
        <= gates["maximum_actual_torque_limit_violation_rate"],
        "nonfinite_rate": summary["nonfinite_rate"]
        <= gates["maximum_nonfinite_rate"],
        "complete_episode_rate": summary["incomplete_episode_rate"]
        <= gates["maximum_incomplete_episode_rate"],
        "torque_saturation_fraction": summary["mean_torque_saturation_fraction"]
        <= gates["maximum_mean_torque_saturation_fraction"],
        "median_direction_error": summary["median_direction_error_deg"]
        <= gates["maximum_median_direction_error_deg"],
        "median_relative_distance_error": summary["median_relative_distance_error"]
        <= gates["maximum_median_relative_distance_error"],
        "worst_direction_bin_success": summary["worst_direction_bin_success_rate"]
        >= gates["minimum_worst_direction_bin_success_rate"],
        "worst_distance_bin_success": summary["worst_distance_bin_success_rate"]
        >= gates["minimum_worst_distance_bin_success_rate"],
    }
    if args.split == "held_out_test":
        gate_checks["passing_fixed_proof_bound_to_same_artifacts"] = fixed_proof is not None
    formal_passed = bool(not args.smoke and all(gate_checks.values()))
    fixed_passed = bool(args.split == "fixed_validation" and formal_passed)
    heldout_passed = bool(args.split == "held_out_test" and formal_passed)
    engineering_smoke_passed = bool(
        args.smoke
        and len(rows) == len(seeds)
        and all(row["steps_executed"] > 0 for row in rows)
        and summary["nonfinite_rate"] == 0.0
    )
    output = _next_output(checkpoint, args.split, args.smoke)
    payload = {
        "schema_version": "v046_strict_pass_eval_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": args.split,
        "formal_protocol": bool(not args.smoke),
        "engineering_smoke": bool(args.smoke),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "acceptance_config": {
            "path": str(contract_path),
            "sha256": _sha256(contract_path),
        },
        "semantic_audit": semantic_audit,
        "fixed_validation_proof": fixed_proof,
        "source_sha256": runtime_sources,
        "source_manifest_sha256": _json_hash(runtime_sources),
        "seed_start": seeds[0],
        "episode_seed_sha256": _json_hash(seeds),
        "scenario_manifest_sha256": _json_hash(scenarios),
        "deterministic_actor": True,
        "device": args.device,
        "summary": summary,
        "direction_bin_evidence": direction_bins,
        "distance_bin_evidence": distance_bins,
        "gate_checks": {name: bool(value) for name, value in gate_checks.items()},
        "engineering_smoke_passed": engineering_smoke_passed,
        "fixed_validation_passed": fixed_passed,
        "heldout_validation_passed": heldout_passed,
        "training_completed": False,
        "video_validation_passed": False,
        # Even a passing held-out trace still needs the separate video proof and
        # complete training record before project-level quality can be promoted.
        "quality_passed": False,
        "episodes": rows,
        "conclusion": (
            "Engineering replay completed; this is not passing-skill evidence."
            if args.smoke
            else (
                f"{args.split} gates passed, but project quality remains false pending video/training proof."
                if formal_passed
                else f"{args.split} failed one or more strict behavior gates."
            )
        ),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "engineering_smoke_passed": engineering_smoke_passed,
                "fixed_validation_passed": fixed_passed,
                "heldout_validation_passed": heldout_passed,
                "quality_passed": False,
                "failed_gates": [name for name, ok in gate_checks.items() if not ok],
            },
            ensure_ascii=False,
        )
    )
    if args.smoke:
        return 0 if engineering_smoke_passed else 1
    return 0 if formal_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
