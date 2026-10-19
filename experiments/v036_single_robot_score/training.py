"""v036 的配置、编号检查点、行为克隆和固定种子评估工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from experiments.v036_single_robot_score.single_robot_score_env import SingleRobotScoreEnv, teacher_action


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "v036_bc_ppo.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iteration(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def list_checkpoints() -> list[Path]:
    return sorted(EXPERIMENT_ROOT.joinpath("checkpoints").glob("v036_policy_iter_*.zip"), key=_iteration)


def latest_checkpoint() -> Path | None:
    checkpoints = list_checkpoints()
    return checkpoints[-1] if checkpoints else None


def next_checkpoint_path() -> Path:
    latest = latest_checkpoint()
    number = 0 if latest is None else _iteration(latest) + 1
    return EXPERIMENT_ROOT / "checkpoints" / f"v036_policy_iter_{number:04d}.zip"


def unique_metrics_path(stem: str) -> Path:
    folder = EXPERIMENT_ROOT / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    candidate = folder / f"{stem}.json"
    run = 2
    while candidate.exists():
        candidate = folder / f"{stem}_run_{run:02d}.json"
        run += 1
    return candidate


def build_ppo(config: dict[str, Any], env: Monitor) -> PPO:
    ppo = config["ppo"]
    return PPO(
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


def collect_teacher_demonstrations(
    config: dict[str, Any], *, sample_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    env = SingleRobotScoreEnv(config, seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    goals = 0
    falls = 0
    obs, _ = env.reset(seed=seed)
    try:
        while len(observations) < sample_count:
            action = teacher_action(obs)
            observations.append(obs.copy())
            actions.append(action.copy())
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                goals += int(bool(info["scored"]))
                falls += int(bool(info["fell"]))
                obs, _ = env.reset()
    finally:
        env.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        {"samples": len(observations), "teacher_goals": goals, "teacher_falls": falls},
    )


def behavior_clone_actor(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    obs_tensor = torch.from_numpy(observations).to(model.device)
    action_tensor = torch.from_numpy(actions).to(model.device)
    losses: list[float] = []
    model.policy.train()
    for _ in range(epochs):
        indexes = rng.permutation(len(observations))
        epoch_losses: list[float] = []
        for start in range(0, len(indexes), batch_size):
            batch = torch.as_tensor(indexes[start : start + batch_size], dtype=torch.long, device=model.device)
            predicted, _, _ = model.policy(obs_tensor[batch], deterministic=True)
            loss = torch.square(predicted - action_tensor[batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return losses


def evaluate_policy(
    policy: PPO | None,
    config: dict[str, Any],
    *,
    use_teacher: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in config["training"]["evaluation_seeds"]:
        env = SingleRobotScoreEnv(config, seed=int(seed))
        observation, _ = env.reset(seed=int(seed))
        total_reward = 0.0
        final_info: dict[str, Any] = {}
        try:
            while True:
                if use_teacher:
                    action = teacher_action(observation)
                else:
                    if policy is None:
                        raise ValueError("policy is required unless use_teacher=True")
                    action, _ = policy.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
        finally:
            env.close()
        rows.append({"seed": int(seed), "episode_reward": total_reward, **final_info})

    goal_count = int(sum(bool(row["scored"]) for row in rows))
    official_goal_count = int(sum(bool(row["official_goal"]) for row in rows))
    fall_count = int(sum(bool(row["fell"]) for row in rows))
    episode_count = len(rows)
    summary = {
        "episodes": episode_count,
        "goal_count": goal_count,
        "official_goal_count": official_goal_count,
        "success_rate": goal_count / episode_count,
        "contact_rate": float(np.mean([bool(row["contacted_ball"]) for row in rows])),
        "fall_rate": fall_count / episode_count,
        "mean_reward": float(np.mean([float(row["episode_reward"]) for row in rows])),
        "mean_minimum_ball_distance_m": float(np.mean([float(row["minimum_ball_distance_m"]) for row in rows])),
        "mean_ball_forward_m": float(np.mean([float(row["ball_forward_m"]) for row in rows])),
        "mean_maximum_ball_speed_mps": float(np.mean([float(row["maximum_ball_speed_mps"]) for row in rows])),
        "minimum_robot_height_m": float(np.min([float(row["minimum_robot_height_m"]) for row in rows])),
        "minimum_robot_upright": float(np.min([float(row["minimum_robot_upright"]) for row in rows])),
        "all_23_targets_sent": bool(all(bool(row["all_23_targets_sent"]) for row in rows)),
    }
    training = config["training"]
    gate_results = {
        "at_least_one_goal": goal_count >= int(training["required_goal_count"]),
        "target_success_rate": summary["success_rate"] >= float(training["target_success_rate"]),
        "fall_rate": summary["fall_rate"] <= float(training["maximum_fall_rate"]),
        "finite_and_23dof": summary["all_23_targets_sent"],
    }
    return {"summary": summary, "gate_results": gate_results, "quality_passed": bool(all(gate_results.values())), "episodes": rows}
