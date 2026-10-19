"""v037 分阶段检查点、质量门和固定种子评估。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from experiments.v037_midfield_dribble_score.midfield_dribble_env import MidfieldDribbleScoreEnv


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "v037_curriculum.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stage_checkpoints(stage_index: int) -> list[Path]:
    return sorted(EXPERIMENT_ROOT.joinpath("checkpoints").glob(f"v037_stage_{stage_index:02d}_iter_*.zip"))


def latest_stage_checkpoint(stage_index: int) -> Path | None:
    paths = stage_checkpoints(stage_index)
    return paths[-1] if paths else None


def next_stage_checkpoint(stage_index: int) -> Path:
    paths = stage_checkpoints(stage_index)
    iteration = 1 if not paths else int(paths[-1].stem.rsplit("_", 1)[-1]) + 1
    return EXPERIMENT_ROOT / "checkpoints" / f"v037_stage_{stage_index:02d}_iter_{iteration:04d}.zip"


def stage_metrics(stage_index: int) -> list[Path]:
    return sorted(EXPERIMENT_ROOT.joinpath("metrics").glob(f"v037_stage_{stage_index:02d}_iter_*_training.json"))


def latest_stage_metrics(stage_index: int) -> tuple[Path | None, dict[str, Any] | None]:
    paths = stage_metrics(stage_index)
    if not paths:
        return None, None
    return paths[-1], json.loads(paths[-1].read_text(encoding="utf-8"))


def source_checkpoint(config: dict[str, Any], stage_index: int) -> Path:
    same_stage = latest_stage_checkpoint(stage_index)
    if same_stage is not None:
        return same_stage
    if stage_index == 0:
        source = PROJECT_ROOT / str(config["bootstrap_model"])
        if not source.is_file():
            raise FileNotFoundError(f"v036 bootstrap model is missing: {source}")
        return source
    previous_metrics_path, previous_metrics = latest_stage_metrics(stage_index - 1)
    if previous_metrics is None or not bool(previous_metrics.get("quality_passed", False)):
        raise RuntimeError(
            f"stage {stage_index - 1} has not passed; latest metrics={previous_metrics_path}. "
            "The curriculum cannot skip a distance stage."
        )
    previous = latest_stage_checkpoint(stage_index - 1)
    if previous is None:
        raise FileNotFoundError(f"stage {stage_index - 1} checkpoint is missing")
    return previous


def unique_metrics_path(stem: str) -> Path:
    folder = EXPERIMENT_ROOT / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.json"
    run = 2
    while path.exists():
        path = folder / f"{stem}_run_{run:02d}.json"
        run += 1
    return path


def evaluate_stage(model: PPO, config: dict[str, Any], stage_index: int) -> dict[str, Any]:
    stage = config["stages"][stage_index]
    rows: list[dict[str, Any]] = []
    for seed in stage["evaluation_seeds"]:
        env = MidfieldDribbleScoreEnv(config, stage_index, seed=int(seed))
        observation, _ = env.reset(seed=int(seed))
        total_reward = 0.0
        final_info: dict[str, Any] = {}
        try:
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
        finally:
            env.close()
        rows.append({"seed": int(seed), "episode_reward": total_reward, **final_info})

    episode_count = len(rows)
    goal_count = int(sum(bool(row["scored"]) for row in rows))
    fall_count = int(sum(bool(row["fell"]) for row in rows))
    summary = {
        "stage_index": stage_index,
        "stage_name": str(stage["name"]),
        "episodes": episode_count,
        "goal_count": goal_count,
        "official_goal_count": int(sum(bool(row["official_goal"]) for row in rows)),
        "success_rate": goal_count / episode_count,
        "contact_rate": float(np.mean([bool(row["contacted_ball"]) for row in rows])),
        "fall_rate": fall_count / episode_count,
        "mean_reward": float(np.mean([float(row["episode_reward"]) for row in rows])),
        "mean_goal_progress_m": float(np.mean([float(row["goal_distance_progress_m"]) for row in rows])),
        "mean_controlled_possession_ratio": float(
            np.mean([float(row["controlled_possession_ratio"]) for row in rows])
        ),
        "mean_max_distance_after_contact_m": float(
            np.mean([float(row["maximum_distance_after_contact_m"]) for row in rows])
        ),
        "lost_possession_rate": float(np.mean([bool(row["lost_possession"]) for row in rows])),
        "mean_episode_seconds": float(np.mean([float(row["episode_seconds"]) for row in rows])),
        "minimum_robot_height_m": float(np.min([float(row["minimum_robot_height_m"]) for row in rows])),
        "minimum_robot_upright": float(np.min([float(row["minimum_robot_upright"]) for row in rows])),
        "all_23_targets_sent": bool(all(bool(row["all_23_targets_sent"]) for row in rows)),
    }
    gates = config["quality_gates"]
    gate_results = {
        "at_least_one_goal": goal_count >= int(gates["required_goal_count"]),
        "target_success_rate": summary["success_rate"] >= float(gates["target_success_rate"]),
        "fall_rate": summary["fall_rate"] <= float(gates["maximum_fall_rate"]),
        "contact_rate": summary["contact_rate"] >= float(gates["minimum_contact_rate"]),
        "controlled_possession": summary["mean_controlled_possession_ratio"]
        >= float(gates["minimum_controlled_possession_ratio"]),
        "finite_and_23dof": summary["all_23_targets_sent"],
    }
    return {
        "stage": stage,
        "summary": summary,
        "gate_results": gate_results,
        "quality_passed": bool(all(gate_results.values())),
        "episodes": rows,
    }
