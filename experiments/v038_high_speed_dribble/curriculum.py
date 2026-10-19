"""v038 编号产物、课程门槛和固定种子评估。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from experiments.v038_high_speed_dribble.high_speed_dribble_env import HighSpeedResidualDribbleEnv


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "v038_speed_curriculum.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iteration(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def stage_checkpoints(stage_index: int) -> list[Path]:
    return sorted(
        EXPERIMENT_ROOT.joinpath("checkpoints").glob(f"v038_stage_{stage_index:02d}_iter_*.zip"),
        key=_iteration,
    )


def latest_stage_checkpoint(stage_index: int) -> Path | None:
    paths = stage_checkpoints(stage_index)
    return paths[-1] if paths else None


def next_stage_checkpoint(stage_index: int) -> Path:
    latest = latest_stage_checkpoint(stage_index)
    iteration = 1 if latest is None else _iteration(latest) + 1
    return EXPERIMENT_ROOT / "checkpoints" / f"v038_stage_{stage_index:02d}_iter_{iteration:04d}.zip"


def unique_metrics_path(stem: str) -> Path:
    folder = EXPERIMENT_ROOT / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    candidate = folder / f"{stem}.json"
    run = 2
    while candidate.exists():
        candidate = folder / f"{stem}_run_{run:02d}.json"
        run += 1
    return candidate


def latest_training_metrics(stage_index: int) -> tuple[Path | None, dict[str, Any] | None]:
    paths = sorted(EXPERIMENT_ROOT.joinpath("metrics").glob(f"v038_stage_{stage_index:02d}_iter_*_training.json"))
    if not paths:
        return None, None
    return paths[-1], json.loads(paths[-1].read_text(encoding="utf-8"))


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_or_continue_model(
    config: dict[str, Any], stage_index: int, env: Monitor
) -> tuple[PPO, str]:
    """同阶段续训；首阶段继承 v037；更高阶段必须先通过上一阶段。"""

    same_stage = latest_stage_checkpoint(stage_index)
    if same_stage is not None:
        return PPO.load(str(same_stage), env=env, device="cpu"), str(same_stage)
    if stage_index == 0:
        bootstrap_eval = _project_path(config["bootstrap_evaluation"])
        bootstrap_payload = json.loads(bootstrap_eval.read_text(encoding="utf-8"))
        if not bool(bootstrap_payload.get("quality_passed", False)):
            raise RuntimeError(f"v037 bootstrap has not passed its recorded gate: {bootstrap_eval}")
        bootstrap = _project_path(config["bootstrap_model"])
        if not bootstrap.is_file():
            raise FileNotFoundError(f"v037 bootstrap checkpoint is missing: {bootstrap}")
        return PPO.load(str(bootstrap), env=env, device="cpu"), str(bootstrap)

    previous_metrics_path, previous_metrics = latest_training_metrics(stage_index - 1)
    if previous_metrics is None or not bool(previous_metrics.get("quality_passed", False)):
        raise RuntimeError(
            f"v038 stage {stage_index - 1} has not passed; latest metrics={previous_metrics_path}"
        )
    previous = latest_stage_checkpoint(stage_index - 1)
    if previous is None:
        raise FileNotFoundError(f"v038 stage {stage_index - 1} checkpoint is missing")
    return PPO.load(str(previous), env=env, device="cpu"), str(previous)


def evaluate_stage(
    model: PPO | None,
    config: dict[str, Any],
    stage_index: int,
    *,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """固定种子独立创建物理环境；model=None 表示纯教师零残差消融。"""

    stage = config["stages"][stage_index]
    evaluation_seeds = [int(seed) for seed in (seeds or stage["evaluation_seeds"])]
    rows: list[dict[str, Any]] = []
    for seed in evaluation_seeds:
        env = HighSpeedResidualDribbleEnv(config, stage_index, seed=seed)
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        final_info: dict[str, Any] = {}
        try:
            while True:
                if model is None:
                    action = np.zeros(3, dtype=np.float32)
                else:
                    action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
        finally:
            env.close()
        rows.append({"seed": seed, "episode_reward": total_reward, **final_info})

    count = len(rows)
    goals = int(sum(bool(row["scored"]) for row in rows))
    summary = {
        "stage_index": int(stage_index),
        "stage_name": str(stage["name"]),
        "episodes": count,
        "goal_count": goals,
        "official_goal_count": int(sum(bool(row["official_goal"]) for row in rows)),
        "success_rate": goals / max(1, count),
        "contact_rate": float(np.mean([bool(row["contacted_ball"]) for row in rows])),
        "fall_rate": float(np.mean([bool(row["fell"]) for row in rows])),
        "lost_possession_rate": float(np.mean([bool(row["lost_possession"]) for row in rows])),
        "mean_goal_progress_m": float(np.mean([float(row["goal_distance_progress_m"]) for row in rows])),
        "mean_controlled_possession_ratio": float(
            np.mean([float(row["controlled_possession_ratio"]) for row in rows])
        ),
        "mean_controlled_robot_speed_mps": float(
            np.mean([float(row["mean_controlled_robot_speed_mps"]) for row in rows])
        ),
        "mean_episode_seconds": float(np.mean([float(row["episode_seconds"]) for row in rows])),
        "all_23_targets_sent": bool(all(bool(row["all_23_targets_sent"]) for row in rows)),
    }
    gates = config["quality_gates"]
    gate_results = {
        "required_goals": goals >= int(gates["required_goal_count"]),
        "target_success_rate": summary["success_rate"] >= float(gates["target_success_rate"]),
        "fall_rate": summary["fall_rate"] <= float(gates["maximum_fall_rate"]),
        "contact_rate": summary["contact_rate"] >= float(gates["minimum_contact_rate"]),
        "controlled_possession": summary["mean_controlled_possession_ratio"]
        >= float(gates["minimum_controlled_possession_ratio"]),
        "lost_possession": summary["lost_possession_rate"]
        <= float(gates["maximum_lost_possession_rate"]),
        "controlled_speed": summary["mean_controlled_robot_speed_mps"]
        >= float(stage["target_controlled_speed_mps"]),
        "finite_and_23dof": summary["all_23_targets_sent"],
    }
    return {
        "method": "high_speed_geometric_teacher_plus_ppo_residual",
        "policy": "zero_residual_teacher" if model is None else "ppo_residual",
        "stage": stage,
        "evaluation_seeds": evaluation_seeds,
        "summary": summary,
        "gate_results": gate_results,
        "quality_passed": bool(all(gate_results.values())),
        "episodes": rows,
    }
