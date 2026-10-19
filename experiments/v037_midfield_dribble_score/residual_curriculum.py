"""v037-r2 教师残差版本的编号检查点、阶段门和评估。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from experiments.v037_midfield_dribble_score.curriculum import DEFAULT_CONFIG, load_config, unique_metrics_path
from experiments.v037_midfield_dribble_score.midfield_dribble_env import ResidualMidfieldDribbleScoreEnv


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def residual_checkpoints(stage_index: int) -> list[Path]:
    return sorted(EXPERIMENT_ROOT.joinpath("checkpoints").glob(f"v037r2_stage_{stage_index:02d}_iter_*.zip"))


def latest_residual_checkpoint(stage_index: int) -> Path | None:
    paths = residual_checkpoints(stage_index)
    return paths[-1] if paths else None


def next_residual_checkpoint(stage_index: int) -> Path:
    paths = residual_checkpoints(stage_index)
    iteration = 1 if not paths else int(paths[-1].stem.rsplit("_", 1)[-1]) + 1
    return EXPERIMENT_ROOT / "checkpoints" / f"v037r2_stage_{stage_index:02d}_iter_{iteration:04d}.zip"


def residual_metrics(stage_index: int) -> list[Path]:
    return sorted(EXPERIMENT_ROOT.joinpath("metrics").glob(f"v037r2_stage_{stage_index:02d}_iter_*_training.json"))


def latest_residual_metrics(stage_index: int) -> tuple[Path | None, dict[str, Any] | None]:
    paths = residual_metrics(stage_index)
    if not paths:
        return None, None
    return paths[-1], json.loads(paths[-1].read_text(encoding="utf-8"))


def build_zero_residual_model(config: dict[str, Any], env: Monitor) -> PPO:
    ppo = config["ppo"]
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(ppo["learning_rate"]),
        n_steps=int(ppo["n_steps"]),
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["n_epochs"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        clip_range=float(ppo["clip_range"]),
        ent_coef=float(ppo["ent_coef"]),
        policy_kwargs={"net_arch": list(ppo["policy_layers"]), "log_std_init": float(ppo["log_std_init"])},
        seed=int(ppo["seed"]),
        device="cpu",
        verbose=1,
    )
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    return model


def load_or_build_stage_model(
    config: dict[str, Any], stage_index: int, env: Monitor
) -> tuple[PPO, str]:
    same_stage = latest_residual_checkpoint(stage_index)
    if same_stage is not None:
        return PPO.load(str(same_stage), env=env, device="cpu"), str(same_stage)
    if stage_index == 0:
        return build_zero_residual_model(config, env), "zero residual on geometric dribble teacher"
    previous_metrics_path, previous_metrics = latest_residual_metrics(stage_index - 1)
    if previous_metrics is None or not bool(previous_metrics.get("quality_passed", False)):
        raise RuntimeError(
            f"residual stage {stage_index - 1} has not passed; latest metrics={previous_metrics_path}"
        )
    previous = latest_residual_checkpoint(stage_index - 1)
    if previous is None:
        raise FileNotFoundError(f"residual stage {stage_index - 1} checkpoint is missing")
    return PPO.load(str(previous), env=env, device="cpu"), str(previous)


def evaluate_residual_stage(model: PPO, config: dict[str, Any], stage_index: int) -> dict[str, Any]:
    stage = config["stages"][stage_index]
    rows: list[dict[str, Any]] = []
    for seed in stage["evaluation_seeds"]:
        env = ResidualMidfieldDribbleScoreEnv(config, stage_index, seed=int(seed))
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

    count = len(rows)
    goals = int(sum(bool(row["scored"]) for row in rows))
    summary = {
        "stage_index": stage_index,
        "stage_name": str(stage["name"]),
        "episodes": count,
        "goal_count": goals,
        "official_goal_count": int(sum(bool(row["official_goal"]) for row in rows)),
        "success_rate": goals / count,
        "contact_rate": float(np.mean([bool(row["contacted_ball"]) for row in rows])),
        "fall_rate": float(np.mean([bool(row["fell"]) for row in rows])),
        "mean_goal_progress_m": float(np.mean([float(row["goal_distance_progress_m"]) for row in rows])),
        "mean_controlled_possession_ratio": float(
            np.mean([float(row["controlled_possession_ratio"]) for row in rows])
        ),
        "lost_possession_rate": float(np.mean([bool(row["lost_possession"]) for row in rows])),
        "mean_episode_seconds": float(np.mean([float(row["episode_seconds"]) for row in rows])),
        "all_23_targets_sent": bool(all(bool(row["all_23_targets_sent"]) for row in rows)),
    }
    gates = config["quality_gates"]
    gate_results = {
        "at_least_one_goal": goals >= int(gates["required_goal_count"]),
        "target_success_rate": summary["success_rate"] >= float(gates["target_success_rate"]),
        "fall_rate": summary["fall_rate"] <= float(gates["maximum_fall_rate"]),
        "contact_rate": summary["contact_rate"] >= float(gates["minimum_contact_rate"]),
        "controlled_possession": summary["mean_controlled_possession_ratio"]
        >= float(gates["minimum_controlled_possession_ratio"]),
        "finite_and_23dof": summary["all_23_targets_sent"],
    }
    return {
        "method": "geometric_dribble_teacher_plus_ppo_residual",
        "stage": stage,
        "summary": summary,
        "gate_results": gate_results,
        "quality_passed": bool(all(gate_results.values())),
        "episodes": rows,
    }
