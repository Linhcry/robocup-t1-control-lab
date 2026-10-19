#!/usr/bin/env python3
"""Strict fixed/held-out v049 evaluation with hash-bound anti-fraud proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from itertools import product
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


def _unique_output(held_out: bool) -> Path:
    stem = "v049_policy_heldout_test" if held_out else "v049_policy_fixed_validation"
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = root / f"{stem}_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v049 evaluation filename")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--fixed-proof", type=Path)
    parser.add_argument("--video-proof", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_json(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _all_true(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


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


def _sources_match(recorded: Any, current: dict[str, dict[str, str]]) -> bool:
    return isinstance(recorded, dict) and all(
        isinstance(recorded.get(name), dict)
        and recorded[name].get("sha256") == item["sha256"]
        for name, item in current.items()
    )


def _valid_audit(proof: dict[str, Any] | None, sources: dict[str, dict[str, str]]) -> bool:
    return bool(
        proof
        and proof.get("schema_version") == "v049_model_semantic_audit_v1"
        and proof.get("model_semantics_passed") is True
        and proof.get("quality_passed") is False
        and _all_true(proof.get("checks"))
        and _sources_match(proof.get("sources"), sources)
    )


def _valid_fixed_proof(
    proof: dict[str, Any] | None,
    checkpoint_sha: str,
    contract_sha: str,
    audit_sha: str,
    sources: dict[str, dict[str, str]],
    required_seeds: list[int],
) -> bool:
    if not proof or not isinstance(proof.get("episodes"), list):
        return False
    try:
        recorded_seeds = [int(row["seed"]) for row in proof["episodes"]]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        proof.get("schema_version") == "v049_strict_evaluation_v1"
        and proof.get("split") == "fixed"
        and proof.get("checkpoint", {}).get("sha256") == checkpoint_sha
        and proof.get("contract", {}).get("sha256") == contract_sha
        and proof.get("semantic_audit", {}).get("sha256") == audit_sha
        and _sources_match(proof.get("sources"), sources)
        and recorded_seeds == required_seeds
        and _all_true(proof.get("common_gate_checks"))
        and _all_true(proof.get("split_gate_checks"))
        and proof.get("fixed_quality_passed") is True
        and proof.get("quality_passed") is False
    )


def _valid_video_proof(proof: dict[str, Any] | None, checkpoint_sha: str) -> bool:
    if not proof or proof.get("schema_version") != "v049_video_grid_proof_v1":
        return False
    if proof.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        return False
    clips = proof.get("clips")
    if not isinstance(clips, list) or len(clips) != 27:
        return False
    expected = set(product(range(3), repeat=3))
    seen: set[tuple[int, int, int]] = set()
    for clip in clips:
        try:
            cell = tuple(int(v) for v in clip["cell"])
            path = Path(clip["path"]).expanduser().resolve()
        except (KeyError, TypeError, ValueError):
            return False
        if len(cell) != 3 or cell not in expected or cell in seen:
            return False
        if not path.is_file() or clip.get("sha256") != _sha256(path):
            return False
        seen.add(cell)
    return seen == expected


def _episode(env, wrapped, policy, seed: int, cell: tuple[int, int, int], horizon: int) -> dict:
    from experiments.v049_fullbody_receive import mdp

    env.event_manager.get_term_cfg("reset_receive_scenario").params["evaluation_cell"] = cell
    wrapped.seed(seed)
    obs, _ = wrapped.reset()
    unexpected_done = False
    for _ in range(horizon):
        with torch.inference_mode():
            action = policy(obs)
        obs, _, done, _ = wrapped.step(action)
        unexpected_done |= bool(done[0])
        if unexpected_done:
            break
    state = mdp.receive_state(env)
    return {
        "seed": seed,
        "cell": list(cell),
        "launch_speed_mps": float(state.launch_speed[0]),
        "launch_angle_deg": float(torch.rad2deg(state.launch_angle_rad[0])),
        "landing_y_m": float(state.target_y[0]),
        "landing_z_m": float(state.target_z[0]),
        "success": bool(state.success[0]),
        "legal_touch": bool(state.legal_touched[0]),
        "illegal_touch": bool(state.illegal_touched[0]),
        "dangerous_ground_contact": bool(state.dangerous_contact[0]),
        "fell": bool(state.fell[0]),
        "joint_violation": bool(state.joint_violation[0]),
        "torque_violation": bool(state.torque_violation[0]),
        "nonfinite": bool(state.nonfinite[0]),
        "missed": bool(state.missed[0]),
        "continuous_control_seconds": float(state.hold_steps[0]) * env.step_dt,
        "speed_reduction_fraction": float(state.speed_reduction[0]),
        "control_distance_m": float(state.ball_distance[0]),
        "unexpected_done": unexpected_done,
        "steps_executed": int(env.episode_length_buf[0]),
    }


def _rate(rows: list[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    audit_path = args.semantic_audit.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    contract_path = EXPERIMENT_ROOT / "configs" / "v049_competitive_acceptance.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checkpoint_sha = _sha256(checkpoint)
    contract_sha = _sha256(contract_path)
    sources = _runtime_sources(contract_path)
    audit = _load_json(audit_path, "semantic audit")
    if not _valid_audit(audit, sources):
        raise ValueError("semantic audit does not match all current v049 source hashes")
    audit_sha = _sha256(audit_path)

    fixed_cfg = contract["fixed_validation"]
    fixed_seeds = list(range(fixed_cfg["seed_start_inclusive"], fixed_cfg["seed_end_exclusive"]))
    split_name = "held_out_test" if args.held_out else "fixed_validation"
    split_cfg = contract[split_name]
    seeds = list(range(split_cfg["seed_start_inclusive"], split_cfg["seed_end_exclusive"]))
    fixed_proof = _load_json(args.fixed_proof, "fixed proof")
    fixed_valid = _valid_fixed_proof(
        fixed_proof, checkpoint_sha, contract_sha, audit_sha, sources, fixed_seeds
    )
    if args.held_out and not fixed_valid:
        raise ValueError("held-out test requires a matching passing fixed proof")
    video_proof = _load_json(args.video_proof, "video proof")
    video_valid = _valid_video_proof(video_proof, checkpoint_sha)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from experiments.v049_fullbody_receive.rcss_mjlab_task import (
        rcss_fullbody_receive_env_cfg,
        rcss_fullbody_receive_ppo_runner_cfg,
    )

    # Evaluation keeps the deployable actor's seeded observation noise and
    # delay.  ``play=True`` would remove corruption for a prettier viewer and
    # would therefore give the policy unrealistically exact ball estimates.
    env_cfg = rcss_fullbody_receive_env_cfg(play=False)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain.num_envs = 1
    env_cfg.episode_length_s = 7.0
    env_cfg.events["reset_receive_scenario"].params["stage_override"] = 3
    env_cfg.terminations["receive_transition"].params["auto_terminate"] = False
    agent_cfg = rcss_fullbody_receive_ppo_runner_cfg()
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
    horizon = round(6.5 / env.step_dt)
    rows = [
        _episode(
            env,
            wrapped,
            policy,
            seed,
            (index % 3, (index // 3) % 3, (index // 9) % 3),
            horizon,
        )
        for index, seed in enumerate(seeds)
    ]
    wrapped.close()

    successes = [row for row in rows if row["success"]]
    cell_rates = {}
    for cell in product(range(3), repeat=3):
        selected = [row for row in rows if tuple(row["cell"]) == cell]
        cell_rates["_".join(map(str, cell))] = _rate(selected, "success")
    summary = {
        "episodes": len(rows),
        "success_rate": _rate(rows, "success"),
        "worst_cell_success_rate": min(cell_rates.values()) if cell_rates else 0.0,
        "legal_touch_given_success_rate": _rate(successes, "legal_touch"),
        "illegal_contact_rate": _rate(rows, "illegal_touch"),
        "dangerous_ground_contact_rate": _rate(rows, "dangerous_ground_contact"),
        "fall_rate": _rate(rows, "fell"),
        "joint_violation_rate": _rate(rows, "joint_violation"),
        "torque_violation_rate": _rate(rows, "torque_violation"),
        "minimum_success_hold_seconds": min(
            (row["continuous_control_seconds"] for row in successes), default=0.0
        ),
        "minimum_success_speed_reduction_fraction": min(
            (row["speed_reduction_fraction"] for row in successes), default=0.0
        ),
        "maximum_success_control_distance_m": max(
            (row["control_distance_m"] for row in successes), default=float("inf")
        ),
    }
    gates = contract["quality_gates"]
    common_checks = {
        "semantic_audit_matches_current_sources": True,
        "exact_episode_count": len(rows) == split_cfg["required_episodes"],
        "all_27_cells_present": len(cell_rates) == 27,
        "legal_touch_given_success": summary["legal_touch_given_success_rate"]
        >= gates["minimum_legal_touch_given_success_rate"],
        "illegal_contact_rate": summary["illegal_contact_rate"] <= gates["maximum_illegal_contact_rate"],
        "dangerous_ground_contact_rate": summary["dangerous_ground_contact_rate"]
        <= gates["maximum_dangerous_ground_contact_rate"],
        "fall_rate": summary["fall_rate"] <= gates["maximum_fall_rate"],
        "joint_violation_rate": summary["joint_violation_rate"]
        <= gates["maximum_joint_violation_rate"],
        "torque_violation_rate": summary["torque_violation_rate"]
        <= gates["maximum_torque_violation_rate"],
        "one_second_hold": summary["minimum_success_hold_seconds"]
        >= gates["minimum_success_hold_seconds"],
        "speed_reduction": summary["minimum_success_speed_reduction_fraction"]
        >= gates["minimum_success_speed_reduction_fraction"],
        "control_radius": summary["maximum_success_control_distance_m"]
        <= gates["maximum_success_control_distance_m"],
    }
    if args.held_out:
        split_checks = {
            "matching_fixed_proof": fixed_valid,
            "heldout_success_rate": summary["success_rate"] >= gates["minimum_heldout_success_rate"],
            "worst_cell_success_rate": summary["worst_cell_success_rate"]
            >= gates["minimum_worst_cell_success_rate"],
            "video_grid_manifest": video_valid,
        }
        fixed_passed = True
        heldout_passed = all(common_checks.values()) and all(split_checks.values())
        quality_passed = heldout_passed
    else:
        split_checks = {
            "fixed_success_rate": summary["success_rate"] >= gates["minimum_fixed_success_rate"],
            "worst_cell_success_rate": summary["worst_cell_success_rate"]
            >= gates["minimum_worst_cell_success_rate"],
        }
        fixed_passed = all(common_checks.values()) and all(split_checks.values())
        heldout_passed = False
        quality_passed = False

    output = _unique_output(args.held_out)
    payload = {
        "schema_version": "v049_strict_evaluation_v1",
        "split": "heldout" if args.held_out else "fixed",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "contract": {"path": str(contract_path), "sha256": contract_sha},
        "semantic_audit": {"path": str(audit_path), "sha256": audit_sha},
        "sources": sources,
        "fixed_proof": str(args.fixed_proof.resolve()) if args.fixed_proof else None,
        "video_proof": str(args.video_proof.resolve()) if args.video_proof else None,
        "deterministic_actor": True,
        "manual_sensor_or_success_mutation": False,
        "summary": summary,
        "cell_success_rates": cell_rates,
        "common_gate_checks": common_checks,
        "split_gate_checks": split_checks,
        "fixed_quality_passed": fixed_passed,
        "heldout_quality_passed": heldout_passed,
        "quality_passed": quality_passed,
        "episodes": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "quality_passed": quality_passed}))
    return 0 if (heldout_passed if args.held_out else fixed_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
