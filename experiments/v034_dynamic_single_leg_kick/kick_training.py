"""Shared configuration, checkpoint, and fixed-seed evaluation helpers for v034."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dynamic_kick_env import DynamicSingleLegKickEnv


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "v034_residual_ppo_r3.json"


class PredictPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_iteration(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def list_checkpoints() -> list[Path]:
    return sorted(
        EXPERIMENT_DIR.joinpath("checkpoints").glob("v034_policy_iter_*.zip"),
        key=checkpoint_iteration,
    )


def latest_checkpoint() -> Path | None:
    checkpoints = list_checkpoints()
    return checkpoints[-1] if checkpoints else None


def next_checkpoint_path() -> Path:
    latest = latest_checkpoint()
    iteration = 1 if latest is None else checkpoint_iteration(latest) + 1
    return EXPERIMENT_DIR / "checkpoints" / f"v034_policy_iter_{iteration:04d}.zip"


def unique_metrics_path(stem: str) -> Path:
    metrics_dir = EXPERIMENT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    candidate = metrics_dir / f"{stem}.json"
    suffix = 2
    while candidate.exists():
        candidate = metrics_dir / f"{stem}_run_{suffix:02d}.json"
        suffix += 1
    return candidate


def run_fixed_evaluation(
    policy: PredictPolicy | None,
    config: dict[str, Any],
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    selected_seeds = seeds or [int(seed) for seed in config["training"]["evaluation_seeds"]]
    rows: list[dict[str, Any]] = []
    env = DynamicSingleLegKickEnv.from_config(config, seed=selected_seeds[0])
    try:
        for seed in selected_seeds:
            observation, _ = env.reset(seed=seed)
            episode_reward = 0.0
            episode_steps = 0
            terminated = truncated = False
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action = (
                    np.zeros(env.action_space.shape, dtype=np.float32)
                    if policy is None
                    else np.asarray(policy.predict(observation, deterministic=True)[0], dtype=np.float32)
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                episode_steps += 1
            rows.append(
                {
                    "seed": seed,
                    "episode_reward": episode_reward,
                    "episode_steps": episode_steps,
                    **info,
                }
            )
    finally:
        env.close()

    recovery_fallback = float(config["environment"]["episode_seconds"])
    recovery_times = [
        recovery_fallback if row["recovery_time_s"] is None else float(row["recovery_time_s"])
        for row in rows
    ]
    summary = {
        "episodes": len(rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "contact_rate": float(np.mean([bool(row["contacted_ball"]) for row in rows])),
        "single_touch_rate": float(
            np.mean([int(row["touch_count"]) == 1 and int(row["left_touch_count"]) == 0 for row in rows])
        ),
        "fall_rate": float(np.mean([bool(row["fallen"]) for row in rows])),
        "invalid_rate": float(np.mean([bool(row["invalid"]) for row in rows])),
        "mean_max_ball_speed_mps": float(np.mean([float(row["max_ball_speed_mps"]) for row in rows])),
        "mean_direction_error_deg": float(np.mean([float(row["direction_error_deg"]) for row in rows])),
        "mean_ball_forward_m": float(np.mean([float(row["ball_forward_m"]) for row in rows])),
        "mean_recovery_time_s": float(np.mean(recovery_times)),
        "max_recovery_time_s": float(np.max(recovery_times)),
        "min_robot_height_m": float(np.min([float(row["min_robot_height_m"]) for row in rows])),
        "min_robot_upright": float(np.min([float(row["min_robot_upright"]) for row in rows])),
        "all_23_targets_sent": bool(all(bool(row["all_23_targets_sent"]) for row in rows)),
    }
    gates = config["quality_gates"]
    gate_results = {
        "success_rate": summary["success_rate"] >= float(gates["success_rate"]),
        "ball_speed_mps": summary["mean_max_ball_speed_mps"] >= float(gates["ball_speed_mps"]),
        "direction_error_deg": summary["mean_direction_error_deg"] <= float(gates["direction_error_deg"]),
        "fall_rate": summary["fall_rate"] <= float(gates["fall_rate"]),
        "single_touch_rate": summary["single_touch_rate"] >= float(gates["single_touch_rate"]),
        "recovery_time_s": summary["max_recovery_time_s"] <= float(gates["recovery_time_s"]),
        "finite_and_23dof": summary["invalid_rate"] == 0.0 and summary["all_23_targets_sent"],
    }
    return {
        "summary": summary,
        "quality_gates": gates,
        "gate_results": gate_results,
        "quality_passed": bool(all(gate_results.values())),
        "episodes": rows,
    }
