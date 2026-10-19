#!/usr/bin/env python3
"""Deterministic fixed or held-out v047 evaluation with fail-closed promotion."""

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
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_output(held_out: bool) -> Path:
    stem = "v047_policy_held_out_test" if held_out else "v047_policy_fixed_validation"
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = root / f"{stem}_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v047 evaluation filename remains")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--target-audit",
        type=Path,
        required=True,
        help="passing v047 target-contract audit for the current task sources",
    )
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument(
        "--fixed-proof",
        type=Path,
        help="required for held-out testing; must pass with the same checkpoint hash",
    )
    parser.add_argument(
        "--video-proof",
        type=Path,
        help="optional immutable 27-cell clip manifest; required for promotion",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_proof(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _all_checks_true(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        item is True for item in value.values()
    )


def _source_hashes_match(
    recorded: Any, expected: dict[str, dict[str, str]]
) -> bool:
    if not isinstance(recorded, dict):
        return False
    return all(
        isinstance(recorded.get(name), dict)
        and recorded[name].get("sha256") == source["sha256"]
        for name, source in expected.items()
    )


def _valid_target_audit(
    proof: dict[str, Any] | None,
    runtime_sources: dict[str, dict[str, str]],
) -> bool:
    if proof is None:
        return False
    return bool(
        proof.get("schema_version") == "v047_target_contract_audit_v1"
        and proof.get("audit_passed") is True
        and proof.get("quality_passed") is False
        and _all_checks_true(proof.get("checks"))
        and _source_hashes_match(proof.get("sources"), runtime_sources)
    )


def _valid_fixed_proof(
    proof: dict[str, Any] | None,
    checkpoint_sha: str,
    contract_sha: str,
    target_audit_sha: str,
    runtime_sources: dict[str, dict[str, str]],
    required_seeds: list[int],
) -> bool:
    if proof is None:
        return False
    rows = proof.get("episodes")
    if not isinstance(rows, list):
        return False
    try:
        recorded_seeds = [int(row["seed"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        proof.get("schema_version") == "v047_strict_evaluation_v1"
        and proof.get("split") == "fixed"
        and proof.get("checkpoint", {}).get("sha256") == checkpoint_sha
        and proof.get("contract", {}).get("sha256") == contract_sha
        and proof.get("target_audit", {}).get("sha256") == target_audit_sha
        and _source_hashes_match(proof.get("sources"), runtime_sources)
        and proof.get("summary", {}).get("episodes") == len(required_seeds)
        and recorded_seeds == required_seeds
        and _all_checks_true(proof.get("common_gate_checks"))
        and _all_checks_true(proof.get("split_gate_checks"))
        and proof.get("fixed_quality_passed") is True
        and proof.get("heldout_quality_passed") is False
        and proof.get("quality_passed") is False
    )


def _valid_video_proof(
    proof: dict[str, Any] | None, checkpoint_sha: str
) -> bool:
    """Require one immutable local clip for every lateral/speed/height cell."""

    if not isinstance(proof, dict):
        return False
    if (
        proof.get("schema_version") != "v047_video_grid_proof_v1"
        or proof.get("checkpoint", {}).get("sha256") != checkpoint_sha
        or proof.get("quality_passed") is not True
    ):
        return False
    clips = proof.get("clips")
    if not isinstance(clips, list) or len(clips) != 27:
        return False
    expected_cells = set(product(range(3), repeat=3))
    seen_cells: set[tuple[int, int, int]] = set()
    for clip in clips:
        if not isinstance(clip, dict):
            return False
        try:
            cell = tuple(int(value) for value in clip["cell"])
            path = Path(clip["path"]).expanduser().resolve()
        except (KeyError, TypeError, ValueError):
            return False
        if len(cell) != 3 or cell not in expected_cells or cell in seen_cells:
            return False
        if not path.is_file() or clip.get("sha256") != _sha256(path):
            return False
        seen_cells.add(cell)
    return seen_cells == expected_cells


def _episode_row(
    env, seed: int, horizon: int, policy, wrapped, cell: tuple[int, int, int]
) -> dict[str, Any]:
    from experiments.v047_fullbody_goalkeeper import mdp

    env.event_manager.get_term_cfg("reset_goalkeeper_scenario").params[
        "evaluation_cell"
    ] = cell
    wrapped.seed(seed)
    obs, _ = wrapped.reset()
    max_reward = -float("inf")
    for _ in range(horizon):
        with torch.inference_mode():
            action = policy(obs)
        obs, reward, done, _ = wrapped.step(action)
        max_reward = max(max_reward, float(reward[0].item()))
        # The evaluation transition is configured not to auto-reset.  Any done
        # here is therefore an unexpected numerical/time-limit failure.
        if bool(done[0]):
            break
    state = mdp.goalkeeper_state(env)
    return {
        "seed": seed,
        "curriculum_stage": int(state.curriculum_stage[0].item()),
        "initial_speed_mps": float(state.initial_speed[0].item()),
        "shot_forward_speed_mps": float(state.shot_forward_speed[0].item()),
        "shot_origin_y_m": float(state.shot_origin_y[0].item()),
        "shot_target_y_m": float(state.shot_target_y[0].item()),
        "shot_target_z_m": float(state.shot_target_z[0].item()),
        "competitive_success": bool(state.success[0].item()),
        "blocked": bool(state.blocked[0].item()),
        "conceded": bool(state.conceded[0].item()),
        "legal_touch": bool(state.ever_legal[0].item()),
        "foot_or_calf_touch": bool(state.ever_foot_or_calf[0].item()),
        "body_touch": bool(state.ever_body[0].item()),
        "head_touch": bool(state.ever_head[0].item()),
        "hand_or_forearm_touch": bool(state.ever_hand[0].item()),
        "peak_head_force_n": float(state.peak_head_force[0].item()),
        "peak_hand_or_forearm_force_n": float(state.peak_hand_force[0].item()),
        "fell": bool(state.fell[0].item()),
        "joint_violation": bool(state.joint_violation[0].item()),
        "unsafe": bool(state.unsafe[0].item()),
        "nonfinite": bool(state.nonfinite[0].item()),
        "torque_saturation_fraction": float(
            state.torque_saturation_steps[0].item() / max(1, horizon)
        ),
        "post_block_stand_seconds": float(
            state.post_block_stand_steps[0].item() * env.step_dt
        ),
        "steps_executed": int(env.episode_length_buf[0].item()),
        "maximum_step_reward": max_reward,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0


def _cell_counts(rows: list[dict[str, Any]], held_cfg: dict[str, Any]) -> dict[str, int]:
    lateral = np.asarray(held_cfg["lateral_bins_m"], dtype=np.float64)
    speed = np.asarray(held_cfg["speed_bins_mps"], dtype=np.float64)
    height = np.asarray(held_cfg["height_bins_m"], dtype=np.float64)
    result: dict[str, int] = {}
    for row in rows:
        li = int(np.clip(np.digitize(row["shot_target_y_m"], lateral) - 1, 0, len(lateral) - 2))
        si = int(np.clip(np.digitize(row["shot_forward_speed_mps"], speed) - 1, 0, len(speed) - 2))
        hi = int(np.clip(np.digitize(row["shot_target_z_m"], height) - 1, 0, len(height) - 2))
        key = f"lateral_{li}_speed_{si}_height_{hi}"
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def main() -> int:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = _sha256(checkpoint)
    contract_path = EXPERIMENT_ROOT / "configs" / "v047_competitive_acceptance.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_sha = _sha256(contract_path)
    runtime_sources = {
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
        "acceptance_contract": {
            "path": str(contract_path),
            "sha256": contract_sha,
        },
    }
    split = contract["held_out_test" if args.held_out else "fixed_validation"]
    seeds = list(range(split["seed_start_inclusive"], split["seed_end_exclusive"]))
    fixed_cfg = contract["fixed_validation"]
    fixed_seeds = list(
        range(fixed_cfg["seed_start_inclusive"], fixed_cfg["seed_end_exclusive"])
    )
    target_audit = _load_proof(args.target_audit, "target audit")
    fixed_proof = _load_proof(args.fixed_proof, "fixed proof")
    video_proof = _load_proof(args.video_proof, "video proof")
    target_audit_path = args.target_audit.resolve()
    target_audit_sha = _sha256(target_audit_path)
    if not _valid_target_audit(target_audit, runtime_sources):
        raise ValueError("target audit does not pass for the current v047 sources")
    fixed_proof_valid = _valid_fixed_proof(
        fixed_proof,
        checkpoint_sha,
        contract_sha,
        target_audit_sha,
        runtime_sources,
        fixed_seeds,
    )
    if args.held_out and fixed_proof is None:
        raise ValueError("held-out testing requires --fixed-proof")
    if args.held_out and not fixed_proof_valid:
        raise ValueError(
            "fixed proof, target audit, sources, contract or checkpoint do not match"
        )
    video_proof_valid = _valid_video_proof(video_proof, checkpoint_sha)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from experiments.v047_fullbody_goalkeeper.rcss_goalkeeper_task import (
        rcss_fullbody_goalkeeper_env_cfg,
        rcss_fullbody_goalkeeper_ppo_runner_cfg,
    )

    env_cfg = rcss_fullbody_goalkeeper_env_cfg(play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain.num_envs = 1
    env_cfg.episode_length_s = 9.0
    env_cfg.events["reset_goalkeeper_scenario"].params["stage_override"] = 3
    env_cfg.terminations["goalkeeper_transition"].params["auto_terminate"] = False
    agent_cfg = rcss_fullbody_goalkeeper_ppo_runner_cfg()
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
    horizon = round(8.0 / env.step_dt)
    rows = []
    for index, seed in enumerate(seeds):
        cell = (index % 3, (index // 3) % 3, (index // 9) % 3)
        rows.append(_episode_row(env, seed, horizon, policy, wrapped, cell))
    wrapped.close()

    summary = {
        "episodes": len(rows),
        "competitive_success_rate": _rate(rows, "competitive_success"),
        "block_rate": _rate(rows, "blocked"),
        "concede_rate": _rate(rows, "conceded"),
        "legal_touch_rate": _rate(rows, "legal_touch"),
        "legal_touch_given_block_rate": float(
            np.mean([row["legal_touch"] for row in rows if row["blocked"]])
        )
        if any(row["blocked"] for row in rows)
        else 0.0,
        "head_danger_rate": float(
            np.mean([row["peak_head_force_n"] > 350.0 for row in rows])
        ),
        "hand_or_forearm_danger_rate": float(
            np.mean([row["peak_hand_or_forearm_force_n"] > 300.0 for row in rows])
        ),
        "fall_rate": _rate(rows, "fell"),
        "joint_violation_rate": _rate(rows, "joint_violation"),
        "nonfinite_rate": _rate(rows, "nonfinite"),
        "mean_torque_saturation_fraction": float(
            np.mean([row["torque_saturation_fraction"] for row in rows])
        ),
        "post_block_standing_rate": float(
            np.mean(
                [
                    row["post_block_stand_seconds"] >= 3.0
                    for row in rows
                    if row["blocked"]
                ]
            )
        )
        if any(row["blocked"] for row in rows)
        else 0.0,
    }
    gates = contract["quality_gates"]
    common_checks = {
        "target_audit_matches_current_sources": True,
        "required_episode_count": len(rows) == split["required_episodes"],
        "legal_touch_given_block": summary["legal_touch_given_block_rate"]
        >= gates["minimum_legal_touch_given_block_rate"],
        "head_danger_rate": summary["head_danger_rate"]
        <= gates["maximum_head_danger_rate"],
        "hand_danger_rate": summary["hand_or_forearm_danger_rate"]
        <= gates["maximum_hand_or_forearm_danger_rate"],
        "fall_rate": summary["fall_rate"] <= gates["maximum_fall_rate"],
        "joint_violation_rate": summary["joint_violation_rate"]
        <= gates["maximum_joint_violation_rate"],
        "nonfinite_rate": summary["nonfinite_rate"] <= gates["maximum_nonfinite_rate"],
        "torque_saturation": summary["mean_torque_saturation_fraction"]
        <= gates["maximum_mean_torque_saturation_fraction"],
        "post_block_standing": summary["post_block_standing_rate"]
        >= gates["minimum_post_block_standing_rate"],
    }
    cell_counts = _cell_counts(rows, contract["held_out_test"])
    if args.held_out:
        split_checks = {
            "fixed_proof_same_checkpoint_contract_sources_and_audit": fixed_proof_valid,
            "heldout_success_rate": summary["competitive_success_rate"]
            >= gates["minimum_held_out_competitive_success_rate"],
            "heldout_block_rate": summary["block_rate"]
            >= gates["minimum_held_out_block_rate"],
            "heldout_concede_rate": summary["concede_rate"]
            <= gates["maximum_held_out_concede_rate"],
            "all_27_grid_cells_present": len(cell_counts) == 27,
            "minimum_per_grid_cell": bool(cell_counts)
            and min(cell_counts.values())
            >= contract["held_out_test"]["minimum_samples_per_lateral_speed_height_cell"],
            "video_grid_proof": video_proof_valid,
        }
        heldout_quality_passed = bool(
            all(common_checks.values()) and all(split_checks.values())
        )
        fixed_quality_passed = True
        quality_passed = heldout_quality_passed
    else:
        split_checks = {
            "fixed_success_rate": summary["competitive_success_rate"]
            >= gates["minimum_fixed_competitive_success_rate"],
        }
        fixed_quality_passed = bool(
            all(common_checks.values()) and all(split_checks.values())
        )
        heldout_quality_passed = False
        # Fixed validation alone never means complete goalkeeper quality.
        quality_passed = False

    output = _unique_output(args.held_out)
    payload = {
        "schema_version": "v047_strict_evaluation_v1",
        "split": "held_out" if args.held_out else "fixed",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "contract": {"path": str(contract_path), "sha256": contract_sha},
        "sources": runtime_sources,
        "target_audit": {
            "path": str(target_audit_path),
            "sha256": target_audit_sha,
        },
        "fixed_proof": (
            {"path": str(args.fixed_proof.resolve()), "sha256": _sha256(args.fixed_proof.resolve())}
            if args.fixed_proof
            else None
        ),
        "video_proof": (
            {"path": str(args.video_proof.resolve()), "sha256": _sha256(args.video_proof.resolve())}
            if args.video_proof
            else None
        ),
        "deterministic_actor": True,
        "summary": summary,
        "grid_cell_counts": cell_counts,
        "common_gate_checks": common_checks,
        "split_gate_checks": split_checks,
        "fixed_quality_passed": fixed_quality_passed,
        "heldout_quality_passed": heldout_quality_passed,
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
                "fixed_quality_passed": fixed_quality_passed,
                "heldout_quality_passed": heldout_quality_passed,
                "quality_passed": quality_passed,
            },
            ensure_ascii=False,
        )
    )
    return 0 if (heldout_quality_passed if args.held_out else fixed_quality_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
