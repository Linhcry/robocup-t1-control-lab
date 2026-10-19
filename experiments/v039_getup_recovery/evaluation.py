"""Fixed-seed evaluation and strict v039 quality gates."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Protocol

import numpy as np

from .getup_env import T1GetUpEnv, load_config


class PredictPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[Any, Any]: ...


def _gate_summary(
    episodes: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gate = config["quality_gate"]
    successes = [episode for episode in episodes if episode["success"]]
    unsafe_count = sum(bool(episode["unsafe"]) for episode in episodes)
    violation_count = sum(bool(episode["joint_limit_violation"]) for episode in episodes)
    sent_target_violation_count = sum(
        bool(episode["sent_target_limit_violation"]) for episode in episodes
    )
    all_23 = all(bool(episode["all_23_targets"]) for episode in episodes)
    by_pose: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_pose[str(episode["pose"])].append(episode)
    pose_success_rates = {
        pose: sum(bool(item["success"]) for item in items) / len(items)
        for pose, items in sorted(by_pose.items())
    }
    success_times = [float(episode["elapsed_s"]) for episode in successes]
    median_time = float(median(success_times)) if success_times else None
    p95_time = (
        float(np.quantile(np.asarray(success_times), 0.95, method="higher"))
        if success_times
        else None
    )
    total = len(episodes)
    success_rate = len(successes) / total if total else 0.0
    unsafe_rate = unsafe_count / total if total else 1.0
    violation_rate = violation_count / total if total else 1.0
    worst_pose = min(pose_success_rates.values()) if pose_success_rates else 0.0
    post_recovery_chain_rate = (
        sum(bool(episode.get("post_recovery_chain_success", False)) for episode in episodes)
        / total
        if total
        else 0.0
    )
    checks = {
        "success_rate": success_rate >= float(gate["minimum_success_rate"]),
        "worst_pose_success_rate": worst_pose
        >= float(gate["minimum_worst_pose_success_rate"]),
        "median_getup_time": median_time is not None
        and median_time <= float(gate["maximum_median_getup_time_s"]),
        "p95_getup_time": p95_time is not None
        and p95_time <= float(gate["maximum_p95_getup_time_s"]),
        "post_recovery_stand_walk_chain": post_recovery_chain_rate
        >= float(gate["minimum_post_recovery_chain_success_rate"]),
        "unsafe_rate": unsafe_rate <= float(gate["maximum_unsafe_rate"]),
        "joint_limit_violation_rate": violation_rate
        <= float(gate["maximum_joint_limit_violation_rate"]),
        "sent_targets_within_limits": sent_target_violation_count == 0,
        "all_23_targets": all_23 if gate["require_all_23_targets"] else True,
    }
    return {
        "episode_count": total,
        "success_count": len(successes),
        "success_rate": success_rate,
        "success_rate_by_pose": pose_success_rates,
        "worst_pose_success_rate": worst_pose,
        "median_getup_time_s": median_time,
        "p95_getup_time_s": p95_time,
        "post_recovery_chain_success_rate": post_recovery_chain_rate,
        "unsafe_rate": unsafe_rate,
        "joint_limit_violation_rate": violation_rate,
        "sent_target_limit_violation_rate": (
            sent_target_violation_count / total if total else 1.0
        ),
        "all_23_targets": all_23,
        "checks": checks,
        "quality_passed": all(checks.values()),
    }


def evaluate(
    policy: PredictPolicy | None,
    *,
    stage: int,
    seeds: list[int],
    config_path: str | Path | None = None,
    zero_action: bool = False,
) -> dict[str, Any]:
    """Evaluate every configured pose against the same seed list."""

    config = load_config(config_path)
    stages = {int(item["stage"]): item for item in config["stages"]}
    stage_cfg = stages[int(stage)]
    episodes: list[dict[str, Any]] = []
    for pose_index, pose in enumerate(stage_cfg["poses"]):
        env = T1GetUpEnv(stage=stage, config_path=config_path, fixed_pose=pose)
        try:
            for seed in seeds:
                actual_seed = int(seed + pose_index * 100_000)
                observation, _ = env.reset(seed=actual_seed)
                total_reward = 0.0
                final_info: dict[str, Any] = {}
                while True:
                    if zero_action:
                        action = np.zeros(23, dtype=np.float32)
                    else:
                        if policy is None:
                            raise ValueError("A policy is required unless zero_action=True")
                        action, _ = policy.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, final_info = env.step(action)
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
                episodes.append(
                    {
                        "seed": actual_seed,
                        "pose": pose,
                        "success": bool(final_info["is_success"]),
                        "unsafe": bool(final_info["unsafe"]),
                        "elapsed_s": float(final_info["elapsed_s"]),
                        "return": total_reward,
                        "height_m": float(final_info["height_m"]),
                        "upright_cos": float(final_info["upright_cos"]),
                        "max_torque_fraction": float(final_info["max_torque_fraction"]),
                        "max_contact_force_n": float(final_info["max_contact_force_n"]),
                        "initial_joint_limit_violation": bool(
                            final_info["initial_joint_limit_violation"]
                        ),
                        "initial_max_joint_overshoot_rad": float(
                            final_info["initial_max_joint_overshoot_rad"]
                        ),
                        "joint_limit_violation": bool(final_info["joint_limit_violation"]),
                        "max_joint_overshoot_rad": float(
                            final_info["max_joint_overshoot_rad"]
                        ),
                        "all_23_targets": bool(final_info["all_23_targets"]),
                        "requested_target_was_clamped": bool(
                            final_info["requested_target_was_clamped"]
                        ),
                        "sent_target_limit_violation": bool(
                            final_info["sent_target_limit_violation"]
                        ),
                        "max_sent_target_overshoot_rad": float(
                            final_info["max_sent_target_overshoot_rad"]
                        ),
                        "finite": bool(final_info["finite"]),
                        "post_recovery_chain_success": False,
                        "post_recovery_chain_reason": (
                            "Recovery did not reach the stable-stand handoff gate."
                            if not final_info["is_success"]
                            else "No validated 23-DoF stand-3s/walk-5s handoff controller is available."
                        ),
                    }
                )
        finally:
            env.close()
    return {
        "experiment": config["experiment"],
        "stage": int(stage),
        "stage_name": stage_cfg["name"],
        "poses": list(stage_cfg["poses"]),
        "seed_count_per_pose": len(seeds),
        "summary": _gate_summary(episodes, config),
        "episodes": episodes,
    }
