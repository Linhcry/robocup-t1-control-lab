#!/usr/bin/env python3
"""Strict vectorized fixed/held-out evaluation for v048 legal tackles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

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


def _read_json(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _all_true(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value
        and all(item is True for item in value.values())
    )


def runtime_control_sources(contract_path: Path) -> dict[str, dict[str, str]]:
    """Hash every local/runtime source that determines policy-side control."""

    from experiments.v048_fullbody_legal_tackle.rcss_mjlab_task import RCSS_T1_XML
    import mjlab

    mjlab_root = Path(mjlab.__file__).resolve().parent
    paths = {
        "v048_task": EXPERIMENT_ROOT / "rcss_mjlab_task.py",
        "v048_safe_actions": EXPERIMENT_ROOT / "safe_actions.py",
        "v048_warmstart": EXPERIMENT_ROOT / "warmstart.py",
        "v048_training_entry": EXPERIMENT_ROOT / "scripts" / "train_v048.py",
        "v048_evaluator": EXPERIMENT_ROOT / "scripts" / "evaluate_v048.py",
        "v048_semantic_audit": EXPERIMENT_ROOT
        / "scripts"
        / "audit_v048_semantics.py",
        "v048_evidence_helper": EXPERIMENT_ROOT / "scripts" / "_evidence.py",
        "t1_23dof_spec": REPO_ROOT / "locomotion" / "t1_spec.py",
        "acceptance_contract_v2": contract_path,
        "installed_rcss_t1_xml": RCSS_T1_XML,
        "mjlab_contact_sensor": mjlab_root / "sensor" / "contact_sensor.py",
        "mjlab_policy_step": mjlab_root / "envs" / "manager_based_rl_env.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"control source files are missing: {missing}")
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
        for name, path in paths.items()
    }


def _sources_match(
    recorded: Any, current: dict[str, dict[str, str]]
) -> bool:
    return bool(
        isinstance(recorded, dict)
        and set(recorded) == set(current)
        and all(
            isinstance(recorded[name], dict)
            and recorded[name].get("sha256") == item["sha256"]
            for name, item in current.items()
        )
    )


def _valid_semantic_audit(
    proof: dict[str, Any] | None, sources: dict[str, dict[str, str]]
) -> bool:
    return bool(
        proof
        and proof.get("schema_version") == "v048_semantic_audit_v2"
        and proof.get("semantic_audit_passed") is True
        and proof.get("quality_passed") is False
        and _all_true(proof.get("checks"))
        and _sources_match(proof.get("sources"), sources)
    )


def _required_seeds(split: dict[str, Any]) -> list[int]:
    seeds = list(
        range(split["seed_start_inclusive"], split["seed_end_exclusive"])
    )
    if len(seeds) != split["required_episodes"]:
        raise ValueError("acceptance split seed range and required episode count differ")
    return seeds


def _valid_fixed_proof(
    proof: dict[str, Any] | None,
    *,
    checkpoint_hash: str,
    contract_hash: str,
    audit_hash: str,
    sources: dict[str, dict[str, str]],
    seeds: list[int],
) -> bool:
    if not proof or not isinstance(proof.get("episodes"), list):
        return False
    try:
        recorded_seeds = [int(row["seed"]) for row in proof["episodes"]]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        proof.get("schema_version") == "v048_strict_evaluation_v2"
        and proof.get("evaluation_split") == "fixed_validation"
        and proof.get("formal_protocol") is True
        and proof.get("engineering_smoke") is False
        and proof.get("checkpoint", {}).get("sha256") == checkpoint_hash
        and proof.get("contract", {}).get("sha256") == contract_hash
        and proof.get("semantic_audit", {}).get("sha256") == audit_hash
        and _sources_match(proof.get("control_sources"), sources)
        and recorded_seeds == seeds
        and _all_true(proof.get("common_gate_checks"))
        and _all_true(proof.get("split_gate_checks"))
        and proof.get("fixed_validation_passed") is True
        and proof.get("heldout_validation_passed") is False
        and proof.get("quality_passed") is False
    )


def _valid_video_proof(
    proof: dict[str, Any] | None,
    *,
    checkpoint_hash: str,
    contract_hash: str,
) -> bool:
    if not proof or proof.get("schema_version") != "v048_video_grid_proof_v1":
        return False
    if proof.get("checkpoint", {}).get("sha256") != checkpoint_hash:
        return False
    if proof.get("contract", {}).get("sha256") != contract_hash:
        return False
    clips = proof.get("clips")
    if not isinstance(clips, list) or len(clips) != 40:
        return False
    expected = {(direction, speed) for direction in range(8) for speed in range(5)}
    seen: set[tuple[int, int]] = set()
    for clip in clips:
        try:
            cell = tuple(int(value) for value in clip["cell"])
            path = Path(clip["path"]).expanduser().resolve()
        except (KeyError, TypeError, ValueError):
            return False
        if len(cell) != 2 or cell not in expected or cell in seen:
            return False
        if not path.is_file() or clip.get("sha256") != _sha256(path):
            return False
        seen.add(cell)
    return seen == expected


def _unique_output(split: str, engineering_smoke: bool) -> Path:
    if engineering_smoke:
        stem = "v048_policy_engineering_smoke"
    elif split == "held_out_test":
        stem = "v048_policy_heldout_test"
    else:
        stem = "v048_policy_fixed_validation"
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = root / f"{stem}_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v048 evaluation filename")


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
    parser.add_argument("--video-proof", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=8)
    parser.add_argument("--smoke-seed", type=int, default=4820000)
    args = parser.parse_args()
    if args.engineering_smoke and args.split != "fixed_validation":
        parser.error("engineering smoke is restricted to the fixed split")
    if args.split == "held_out_test" and args.fixed_proof is None:
        parser.error("held-out evaluation requires --fixed-proof")
    if args.fixed_proof is not None and args.split != "held_out_test":
        parser.error("--fixed-proof is only valid for held-out evaluation")
    if args.video_proof is not None and args.split != "held_out_test":
        parser.error("--video-proof is only valid for held-out evaluation")
    if args.smoke_steps <= 0:
        parser.error("smoke-steps must be positive")
    return args


def _scenario_schedule(
    seeds: list[int], contract: dict[str, Any]
) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
    grid = contract["scenario_grid"]
    direction_edges = grid["direction_edges_rad"]
    speed_edges = grid["speed_edges_mps"]
    direction_bins: list[int] = []
    speed_bins: list[int] = []
    directions: list[float] = []
    speeds: list[float] = []
    lateral_offsets: list[float] = []
    for index, seed in enumerate(seeds):
        direction_bin = index % 8
        speed_bin = (index // 8) % 5
        direction_bins.append(direction_bin)
        speed_bins.append(speed_bin)
        directions.append(
            0.5
            * (
                float(direction_edges[direction_bin])
                + float(direction_edges[direction_bin + 1])
            )
        )
        speeds.append(
            0.5
            * (
                float(speed_edges[speed_bin])
                + float(speed_edges[speed_bin + 1])
            )
        )
        rng = np.random.default_rng(seed)
        lateral_offsets.append(float(rng.uniform(-0.35, 0.35)))
    return direction_bins, speed_bins, directions, speeds, lateral_offsets


def _bool_list(value: torch.Tensor) -> list[bool]:
    return [bool(item) for item in value.detach().cpu().tolist()]


def _float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().tolist()]


def _run_vectorized_episodes(
    checkpoint: Path,
    device: str,
    seeds: list[int],
    contract: dict[str, Any],
    *,
    steps: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from experiments.v048_fullbody_legal_tackle.rcss_mjlab_task import (
        EFFORT_LIMITS,
        RCSS_JOINT_NAMES,
        exact_rcss_tackle_env_cfg,
        v048_ppo_runner_cfg,
    )

    (
        direction_bins,
        speed_bins,
        directions,
        speeds,
        lateral_offsets,
    ) = _scenario_schedule(seeds, contract)
    env_cfg = exact_rcss_tackle_env_cfg(play=True)
    env_cfg.scene.num_envs = len(seeds)
    env_cfg.episode_length_s = 8.5
    # Keep tracker side effects but disable automatic reset so final cumulative
    # evidence always belongs to the original episode, including failures.
    env_cfg.terminations = {
        "tracker_update": env_cfg.terminations["tracker_update"],
    }
    reset_cfg = env_cfg.events["correlated_tackle_reset"]
    reset_cfg.params.update(
        {
            "evaluation_direction_rad": directions,
            "evaluation_speed_mps": speeds,
            "evaluation_lateral_offset_m": lateral_offsets,
        }
    )
    agent_cfg = v048_ppo_runner_cfg()
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    wrapped.seed(seeds[0])
    observations, _ = wrapped.reset()
    horizon = (
        int(steps)
        if steps is not None
        else round(float(contract["scenario_grid"]["episode_seconds"]) / env.step_dt)
    )
    unexpected_done = torch.zeros(len(seeds), dtype=torch.bool, device=env.device)
    executed_steps = 0
    evaluation_exception: str | None = None
    try:
        for _ in range(horizon):
            with torch.inference_mode():
                actions = policy(observations)
            observations, _, dones, _ = wrapped.step(actions)
            unexpected_done |= dones.bool()
            executed_steps += 1
    except Exception as error:  # evidence must fail closed and retain the error
        evaluation_exception = f"{type(error).__name__}: {error}"

    attacker_first = _bool_list(env._v048_attacker_first_touch)
    first_touch_owner = env._v048_first_touch_owner.detach().cpu().tolist()
    defender_touched = _bool_list(env._v048_defender_touched)
    turnover = _bool_list(env._v048_turnover)
    success = _bool_list(env._v048_success)
    upper_foul = _bool_list(env._v048_upper_foul)
    lower_foul = _bool_list(env._v048_lower_foul)
    attacker_fell = _bool_list(env._v048_attacker_fell)
    defender_fell = _bool_list(env._v048_defender_fell)
    attacker_joint = _bool_list(env._v048_attacker_joint_violation)
    defender_joint = _bool_list(env._v048_defender_joint_violation)
    attacker_torque = _bool_list(env._v048_attacker_torque_violation)
    defender_torque = _bool_list(env._v048_defender_torque_violation)
    nonfinite = _bool_list(env._v048_nonfinite)
    clearance = _float_list(env._v048_max_clearance)
    control_seconds = _float_list(env._v048_max_control_steps.float() * env.step_dt)
    cross_steps = env._v048_cross_contact_steps.detach().cpu().tolist()
    simultaneous_steps = env._v048_simultaneous_contact_steps.detach().cpu().tolist()
    max_opponent_force = _float_list(env._v048_max_opponent_force)
    attacker_max_torque = env._v048_attacker_max_abs_torque.detach().cpu()
    defender_max_torque = env._v048_defender_max_abs_torque.detach().cpu()
    torque_limits = EFFORT_LIMITS.detach().cpu()
    done_list = _bool_list(unexpected_done)
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        attacker_ratios = attacker_max_torque[index] / torque_limits
        defender_ratios = defender_max_torque[index] / torque_limits
        rows.append(
            {
                "seed": seed,
                "direction_bin": direction_bins[index],
                "speed_bin": speed_bins[index],
                "direction_rad": directions[index],
                "speed_mps": speeds[index],
                "lateral_offset_m": lateral_offsets[index],
                "first_touch_owner": int(first_touch_owner[index]),
                "attacker_real_legal_first_touch": attacker_first[index],
                "defender_real_legal_touch": defender_touched[index],
                "attacker_then_defender_transfer": turnover[index],
                "simultaneous_contact_steps": int(simultaneous_steps[index]),
                "maximum_clearance_m": clearance[index],
                "maximum_continuous_control_seconds": control_seconds[index],
                "cross_robot_contact_seconds": float(cross_steps[index])
                * env.step_dt,
                "maximum_cross_robot_force_n": max_opponent_force[index],
                "upper_body_foul": upper_foul[index],
                "lower_body_foul": lower_foul[index],
                "attacker_fell": attacker_fell[index],
                "defender_fell": defender_fell[index],
                "attacker_joint_violation": attacker_joint[index],
                "defender_joint_violation": defender_joint[index],
                "attacker_actual_torque_violation": attacker_torque[index],
                "defender_actual_torque_violation": defender_torque[index],
                "nonfinite": nonfinite[index],
                "attacker_max_abs_actual_torque_nm_by_joint": {
                    name: float(value)
                    for name, value in zip(
                        RCSS_JOINT_NAMES,
                        attacker_max_torque[index].tolist(),
                        strict=True,
                    )
                },
                "defender_max_abs_actual_torque_nm_by_joint": {
                    name: float(value)
                    for name, value in zip(
                        RCSS_JOINT_NAMES,
                        defender_max_torque[index].tolist(),
                        strict=True,
                    )
                },
                "attacker_max_actual_torque_ratio": float(attacker_ratios.max()),
                "defender_max_actual_torque_ratio": float(defender_ratios.max()),
                "unexpected_termination": done_list[index],
                "steps_executed": executed_steps,
                "success": success[index],
            }
        )
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
    runtime = {
        "device": device,
        "num_parallel_environments": len(seeds),
        "policy_steps_requested": horizon,
        "policy_steps_executed": executed_steps,
        "physics_substeps_per_policy_step": env.cfg.decimation,
        "contact_force_history_shapes": contact_history_shapes,
        "evaluation_exception": evaluation_exception,
    }
    wrapped.close()
    return rows, runtime


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0


def _bin_evidence(
    rows: list[dict[str, Any]], key: str, bins: int
) -> tuple[dict[str, dict[str, float | int]], float]:
    evidence: dict[str, dict[str, float | int]] = {}
    for index in range(bins):
        selected = [row for row in rows if int(row[key]) == index]
        evidence[str(index)] = {
            "episodes": len(selected),
            "success_rate": _rate(selected, "success"),
        }
    worst = min(float(item["success_rate"]) for item in evidence.values())
    return evidence, worst


def _cell_evidence(
    rows: list[dict[str, Any]], samples_per_cell: int
) -> tuple[dict[str, dict[str, float | int]], float, bool]:
    evidence: dict[str, dict[str, float | int]] = {}
    counts_exact = True
    for direction in range(8):
        for speed in range(5):
            selected = [
                row
                for row in rows
                if row["direction_bin"] == direction and row["speed_bin"] == speed
            ]
            evidence[f"direction_{direction}_speed_{speed}"] = {
                "episodes": len(selected),
                "success_rate": _rate(selected, "success"),
            }
            counts_exact &= len(selected) == samples_per_cell
    worst = min(float(item["success_rate"]) for item in evidence.values())
    return evidence, worst, counts_exact


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    audit_path = args.semantic_audit.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    contract_path = (
        EXPERIMENT_ROOT / "configs" / "v048_competitive_acceptance_v2.json"
    ).resolve()
    contract = _read_json(contract_path, "v048 acceptance contract")
    assert contract is not None
    checkpoint_hash = _sha256(checkpoint)
    contract_hash = _sha256(contract_path)
    sources = runtime_control_sources(contract_path)
    audit = _read_json(audit_path, "v048 semantic audit")
    if not _valid_semantic_audit(audit, sources):
        raise ValueError(
            "semantic audit is stale, incomplete, or does not bind every current "
            "v048 control source"
        )
    audit_hash = _sha256(audit_path)

    fixed_seeds = _required_seeds(contract["fixed_validation"])
    split_cfg = contract[args.split]
    seeds = (
        [args.smoke_seed]
        if args.engineering_smoke
        else _required_seeds(split_cfg)
    )
    fixed_proof = _read_json(args.fixed_proof, "fixed proof")
    fixed_proof_valid = _valid_fixed_proof(
        fixed_proof,
        checkpoint_hash=checkpoint_hash,
        contract_hash=contract_hash,
        audit_hash=audit_hash,
        sources=sources,
        seeds=fixed_seeds,
    )
    if args.split == "held_out_test" and not fixed_proof_valid:
        raise ValueError(
            "held-out test requires the same checkpoint's current passing fixed proof"
        )
    video_proof = _read_json(args.video_proof, "video proof")
    video_proof_valid = _valid_video_proof(
        video_proof,
        checkpoint_hash=checkpoint_hash,
        contract_hash=contract_hash,
    )

    rows, runtime = _run_vectorized_episodes(
        checkpoint,
        args.device,
        seeds,
        contract,
        steps=args.smoke_steps if args.engineering_smoke else None,
    )
    successful = [row for row in rows if row["success"]]
    direction_evidence, worst_direction = _bin_evidence(rows, "direction_bin", 8)
    speed_evidence, worst_speed = _bin_evidence(rows, "speed_bin", 5)
    cell_evidence, worst_cell, exact_cell_counts = _cell_evidence(
        rows,
        0
        if args.engineering_smoke
        else int(split_cfg["samples_per_direction_speed_cell"]),
    )
    joint_violation_rate = float(
        np.mean(
            [
                row["attacker_joint_violation"]
                or row["defender_joint_violation"]
                for row in rows
            ]
        )
    )
    torque_violation_rate = float(
        np.mean(
            [
                row["attacker_actual_torque_violation"]
                or row["defender_actual_torque_violation"]
                for row in rows
            ]
        )
    )
    summary = {
        "episodes": len(rows),
        "success_rate": _rate(rows, "success"),
        "attacker_first_touch_rate": _rate(
            rows, "attacker_real_legal_first_touch"
        ),
        "real_transfer_rate": _rate(rows, "attacker_then_defender_transfer"),
        "real_transfer_given_success_rate": _rate(
            successful, "attacker_then_defender_transfer"
        ),
        "minimum_clearance_given_success_m": min(
            (float(row["maximum_clearance_m"]) for row in successful), default=0.0
        ),
        "minimum_control_given_success_s": min(
            (
                float(row["maximum_continuous_control_seconds"])
                for row in successful
            ),
            default=0.0,
        ),
        "upper_body_foul_rate": _rate(rows, "upper_body_foul"),
        "lower_body_foul_rate": _rate(rows, "lower_body_foul"),
        "attacker_fall_rate": _rate(rows, "attacker_fell"),
        "defender_fall_rate": _rate(rows, "defender_fell"),
        "joint_violation_rate": joint_violation_rate,
        "actual_torque_violation_rate": torque_violation_rate,
        "nonfinite_rate": _rate(rows, "nonfinite"),
        "unexpected_termination_rate": _rate(rows, "unexpected_termination"),
        "robot_collision_rate": float(
            np.mean([row["cross_robot_contact_seconds"] > 0.0 for row in rows])
        ),
        "maximum_cross_robot_force_n": max(
            (float(row["maximum_cross_robot_force_n"]) for row in rows),
            default=0.0,
        ),
        "worst_direction_success_rate": worst_direction,
        "worst_speed_success_rate": worst_speed,
        "worst_direction_speed_cell_success_rate": worst_cell,
    }
    common = contract["common_quality_gates"]
    common_checks = {
        "semantic_audit_matches_all_current_sources": True,
        "evaluation_completed_without_exception": runtime["evaluation_exception"]
        is None,
        "four_physics_substeps_per_action": runtime[
            "physics_substeps_per_policy_step"
        ]
        == 4,
        "all_contact_histories_cover_four_substeps": all(
            shape[2] == 4
            for shape in runtime["contact_force_history_shapes"].values()
        ),
        "exact_episode_count": len(rows) == split_cfg["required_episodes"],
        "all_40_cells_have_exact_samples": exact_cell_counts,
        "attacker_real_first_touch_rate": summary["attacker_first_touch_rate"]
        >= common["minimum_attacker_first_touch_rate"],
        "real_transfer_given_success": summary["real_transfer_given_success_rate"]
        >= common["minimum_real_transfer_given_success_rate"],
        "clearance_given_success": summary["minimum_clearance_given_success_m"]
        >= common["minimum_clearance_given_success_m"],
        "two_second_control_given_success": summary[
            "minimum_control_given_success_s"
        ]
        >= common["minimum_control_given_success_s"],
        "upper_body_foul_rate": summary["upper_body_foul_rate"]
        <= common["maximum_upper_body_foul_rate"],
        "lower_body_foul_rate": summary["lower_body_foul_rate"]
        <= common["maximum_lower_body_foul_rate"],
        "attacker_fall_rate": summary["attacker_fall_rate"]
        <= common["maximum_attacker_fall_rate"],
        "defender_fall_rate": summary["defender_fall_rate"]
        <= common["maximum_defender_fall_rate"],
        "joint_violation_rate": summary["joint_violation_rate"]
        <= common["maximum_joint_violation_rate"],
        "actual_torque_violation_rate": summary["actual_torque_violation_rate"]
        <= common["maximum_actual_torque_violation_rate"],
        "nonfinite_rate": summary["nonfinite_rate"]
        <= common["maximum_nonfinite_rate"],
        "unexpected_termination_rate": summary["unexpected_termination_rate"]
        <= common["maximum_unexpected_termination_rate"],
    }
    split_checks = {
        "global_success_rate": summary["success_rate"]
        >= split_cfg["minimum_success_rate"],
        "worst_direction_success_rate": summary["worst_direction_success_rate"]
        >= split_cfg["minimum_worst_direction_success_rate"],
        "worst_speed_success_rate": summary["worst_speed_success_rate"]
        >= split_cfg["minimum_worst_speed_success_rate"],
        "worst_direction_speed_cell_success_rate": summary[
            "worst_direction_speed_cell_success_rate"
        ]
        >= split_cfg["minimum_worst_cell_success_rate"],
    }
    if args.split == "held_out_test":
        split_checks["same_checkpoint_passing_fixed_proof"] = fixed_proof_valid
        split_checks["forty_cell_video_manifest"] = video_proof_valid

    formal_passed = bool(
        not args.engineering_smoke
        and all(common_checks.values())
        and all(split_checks.values())
    )
    fixed_passed = bool(args.split == "fixed_validation" and formal_passed)
    heldout_passed = bool(args.split == "held_out_test" and formal_passed)
    quality_passed = heldout_passed
    engineering_smoke_passed = bool(
        args.engineering_smoke
        and runtime["evaluation_exception"] is None
        and len(rows) == 1
        and runtime["policy_steps_executed"] == args.smoke_steps
        and runtime["physics_substeps_per_policy_step"] == 4
        and all(
            shape[2] == 4
            for shape in runtime["contact_force_history_shapes"].values()
        )
    )
    output = _unique_output(args.split, args.engineering_smoke)
    payload = {
        "schema_version": "v048_strict_evaluation_v2",
        "evaluation_split": args.split,
        "formal_protocol": not args.engineering_smoke,
        "engineering_smoke": bool(args.engineering_smoke),
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash},
        "contract": {"path": str(contract_path), "sha256": contract_hash},
        "semantic_audit": {"path": str(audit_path), "sha256": audit_hash},
        "control_sources": sources,
        "fixed_validation_proof": (
            {
                "path": str(args.fixed_proof.resolve()),
                "sha256": _sha256(args.fixed_proof.resolve()),
                "valid_for_same_checkpoint_and_sources": fixed_proof_valid,
            }
            if args.fixed_proof
            else None
        ),
        "video_proof": (
            {
                "path": str(args.video_proof.resolve()),
                "sha256": _sha256(args.video_proof.resolve()),
                "valid_40_cell_manifest": video_proof_valid,
            }
            if args.video_proof
            else None
        ),
        "deterministic_actor": True,
        "manual_contact_success_or_safety_mutation": False,
        "auto_reset_disabled_for_evidence_integrity": True,
        "runtime": runtime,
        "summary": summary,
        "direction_bin_evidence": direction_evidence,
        "speed_bin_evidence": speed_evidence,
        "direction_speed_cell_evidence": cell_evidence,
        "common_gate_checks": common_checks,
        "split_gate_checks": split_checks,
        "engineering_smoke_passed": engineering_smoke_passed,
        "fixed_validation_passed": fixed_passed,
        "heldout_validation_passed": heldout_passed,
        "quality_passed": quality_passed,
        "episodes": rows,
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
                "quality_passed": quality_passed,
            },
            ensure_ascii=False,
        )
    )
    if args.engineering_smoke:
        return 0 if engineering_smoke_passed else 1
    return 0 if formal_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
