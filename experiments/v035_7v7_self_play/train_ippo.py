"""参数共享 IPPO + GAE + 对手快照的 7v7 自博弈训练。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from experiments.v035_7v7_self_play.artifacts import next_numbered_path
from experiments.v035_7v7_self_play.gate_status import require_v035_training_gate
from experiments.v035_7v7_self_play.local_tactics import ACTION_DIM, LOCAL_OBS_DIM, heuristic_local_action
from experiments.v035_7v7_self_play.match_7v7 import DEFAULT_LOW_LEVEL_MODEL
from experiments.v035_7v7_self_play.networks import (
    SharedActorCritic,
    load_tactical_checkpoint,
    save_tactical_checkpoint,
)
from experiments.v035_7v7_self_play.self_play_env import TacticalSelfPlayEnv


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def opponent_actions(model: SharedActorCritic | None, observations: np.ndarray) -> np.ndarray:
    """训练初期使用启发式，之后使用冻结的旧 Actor 快照。"""

    if model is None:
        return np.stack([heuristic_local_action(row) for row in observations])
    with torch.inference_mode():
        tensor = torch.from_numpy(observations)
        return model.deterministic_actions(tensor).cpu().numpy().astype(np.float32)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """沿时间反向计算七条球员轨迹的广义优势估计。"""

    advantages = np.zeros_like(rewards, dtype=np.float32)
    running_advantage = np.zeros(7, dtype=np.float32)
    next_values = last_values.astype(np.float32)
    for step in range(len(rewards) - 1, -1, -1):
        not_done = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * not_done - values[step]
        running_advantage = delta + gamma * gae_lambda * not_done * running_advantage
        advantages[step] = running_advantage
        next_values = values[step]
    returns = advantages + values
    return advantages, returns


def main() -> None:
    """收集局部轨迹、执行 PPO 裁剪更新，并定期冻结对手快照。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--snapshot-interval", type=int, default=5)
    parser.add_argument("--resume", type=Path, help="通常先加载行为克隆模型")
    parser.add_argument("--low-level-model", type=Path, default=DEFAULT_LOW_LEVEL_MODEL)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    require_v035_training_gate()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SharedActorCritic() if args.resume is None else load_tactical_checkpoint(args.resume)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1.0e-5)
    env = TacticalSelfPlayEnv(args.low_level_model, seed=args.seed)
    left_observations, right_observations = env.reset()
    frozen_opponent: SharedActorCritic | None = None

    for update in range(1, args.updates + 1):
        observation_buffer = np.zeros((args.rollout_steps, 7, LOCAL_OBS_DIM), dtype=np.float32)
        action_buffer = np.zeros((args.rollout_steps, 7, ACTION_DIM), dtype=np.float32)
        log_prob_buffer = np.zeros((args.rollout_steps, 7), dtype=np.float32)
        value_buffer = np.zeros((args.rollout_steps, 7), dtype=np.float32)
        reward_buffer = np.zeros((args.rollout_steps, 7), dtype=np.float32)
        done_buffer = np.zeros((args.rollout_steps, 7), dtype=np.float32)
        rollout_infos: list[dict[str, object]] = []

        for step in range(args.rollout_steps):
            observation_tensor = torch.from_numpy(left_observations)
            with torch.no_grad():
                means, log_stds, values = model(observation_tensor)
                distribution = Normal(means, log_stds.exp())
                raw_actions = distribution.rsample()
                actions = torch.tanh(raw_actions)
                log_probabilities = (
                    distribution.log_prob(raw_actions) - torch.log(1.0 - actions.square() + 1.0e-6)
                ).sum(dim=-1)

            right_actions = opponent_actions(frozen_opponent, right_observations)
            next_left, next_right, rewards, done, info = env.step(actions.numpy(), right_actions)
            observation_buffer[step] = left_observations
            action_buffer[step] = actions.numpy()
            log_prob_buffer[step] = log_probabilities.numpy()
            value_buffer[step] = values.numpy()
            reward_buffer[step] = rewards
            done_buffer[step] = float(done)
            rollout_infos.append(info)
            if done:
                left_observations, right_observations = env.reset()
            else:
                left_observations, right_observations = next_left, next_right

        with torch.no_grad():
            _, _, last_value_tensor = model(torch.from_numpy(left_observations))
        advantages, returns = compute_gae(
            reward_buffer,
            value_buffer,
            done_buffer,
            last_value_tensor.numpy(),
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        flat_observations = torch.from_numpy(observation_buffer.reshape(-1, LOCAL_OBS_DIM))
        flat_actions = torch.from_numpy(action_buffer.reshape(-1, ACTION_DIM))
        flat_old_log_probs = torch.from_numpy(log_prob_buffer.reshape(-1))
        flat_returns = torch.from_numpy(returns.reshape(-1))
        flat_advantages = torch.from_numpy(advantages.reshape(-1))
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1.0e-8)

        losses: list[float] = []
        sample_count = len(flat_observations)
        for _ in range(args.epochs):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, args.minibatch_size):
                indexes = permutation[start : start + args.minibatch_size]
                means, log_stds, values = model(flat_observations[indexes])
                distribution = Normal(means, log_stds.exp())
                safe_actions = torch.clamp(flat_actions[indexes], -0.999999, 0.999999)
                raw_actions = torch.atanh(safe_actions)
                new_log_probs = (
                    distribution.log_prob(raw_actions) - torch.log(1.0 - safe_actions.square() + 1.0e-6)
                ).sum(dim=-1)
                ratios = torch.exp(new_log_probs - flat_old_log_probs[indexes])
                unclipped = ratios * flat_advantages[indexes]
                clipped = torch.clamp(ratios, 1.0 - args.clip_range, 1.0 + args.clip_range) * flat_advantages[indexes]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.square(values - flat_returns[indexes]).mean()
                entropy = distribution.entropy().sum(dim=-1).mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                losses.append(float(loss.detach()))

        if update % args.snapshot_interval == 0:
            frozen_opponent = deepcopy(model).eval()
        checkpoint = next_numbered_path(EXPERIMENT_ROOT / "checkpoints", "v035_ippo", ".pt")
        metrics = next_numbered_path(EXPERIMENT_ROOT / "metrics", "v035_ippo", ".json")
        save_tactical_checkpoint(checkpoint, model, training_method="parameter_sharing_ippo", update=update)
        payload = {
            "update": update,
            "environment_steps": update * args.rollout_steps,
            "player_samples": update * args.rollout_steps * 7,
            "mean_reward": float(reward_buffer.mean()),
            "mean_loss": float(np.mean(losses)),
            "mean_ball_progress_m": float(np.mean([float(info["ball_progress_m"]) for info in rollout_infos])),
            "mean_fallen_players": float(np.mean([float(info["fallen_players"]) for info in rollout_infos])),
            "opponent_snapshot_active": frozen_opponent is not None,
            "checkpoint": str(checkpoint),
        }
        metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**payload, "metrics": str(metrics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
