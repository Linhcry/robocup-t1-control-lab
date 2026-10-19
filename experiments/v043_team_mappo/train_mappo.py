"""Train the v043 recurrent actor with a centralized critic and snapshot league."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from experiments.v043_team_mappo.artifacts import next_numbered_path
from experiments.v043_team_mappo.environment import TeamSelfPlayEnv
from experiments.v043_team_mappo.gate_status import require_team_training_gate
from experiments.v043_team_mappo.networks import RecurrentMAPPO, load_checkpoint, save_checkpoint
from experiments.v043_team_mappo.opponent_pool import OpponentController, OpponentPool
from experiments.v043_team_mappo.spec import ACTION_DIM, CENTRAL_STATE_DIM, LOCAL_OBSERVATION_DIM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOW_LEVEL_MODEL = PROJECT_ROOT.parents[1] / "third_party" / "booster_gym" / "deploy" / "models" / "T1.pt"


def squashed_log_prob(distribution: Normal, raw_actions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return (
        distribution.log_prob(raw_actions) - torch.log(1.0 - actions.square() + 1.0e-6)
    ).sum(dim=-1)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros(rewards.shape[1], dtype=np.float32)
    next_values = last_values.astype(np.float32)
    for step in range(len(rewards) - 1, -1, -1):
        not_done = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * next_values * not_done - values[step]
        running = delta + gamma * gae_lambda * not_done * running
        advantages[step] = running
        next_values = values[step]
    return advantages, advantages + values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players-per-team", type=int, choices=range(3, 8), default=3)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.10)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--snapshot-interval", type=int, default=5)
    parser.add_argument("--teacher-probability", type=float, default=0.40)
    parser.add_argument("--episode-high-level-steps", type=int, default=300)
    parser.add_argument("--prepare-seconds", type=float, default=0.5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--low-level-model", type=Path, default=DEFAULT_LOW_LEVEL_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=43001)
    args = parser.parse_args()
    require_team_training_gate()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if args.resume is None:
        model = RecurrentMAPPO().to(device)
        start_update = 0
    else:
        model, resume_payload = load_checkpoint(args.resume, device=device)
        start_update = int(resume_payload.get("update", 0))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1.0e-5)
    if args.resume is not None and "optimizer_state_dict" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    env = TeamSelfPlayEnv(
        args.low_level_model,
        players_per_team=args.players_per_team,
        episode_high_level_steps=args.episode_high_level_steps,
        prepare_seconds=args.prepare_seconds,
        seed=args.seed,
    )
    pool = OpponentPool(
        EXPERIMENT_ROOT / "opponents_local_v2",
        teacher_probability=args.teacher_probability,
        seed=args.seed + 1,
    )
    left_view, right_view, _ = env.reset()
    hidden = model.initial_hidden(args.players_per_team, device=device)
    episode_start = True
    opponent = OpponentController(pool.sample(), args.players_per_team, device=args.device)

    for relative_update in range(1, args.updates + 1):
        update = start_update + relative_update
        initial_hidden = hidden.detach().clone()
        observations = np.zeros(
            (args.rollout_steps, args.players_per_team, LOCAL_OBSERVATION_DIM), dtype=np.float32
        )
        central_states = np.zeros((args.rollout_steps, CENTRAL_STATE_DIM), dtype=np.float32)
        actions_buffer = np.zeros(
            (args.rollout_steps, args.players_per_team, ACTION_DIM), dtype=np.float32
        )
        log_probs = np.zeros((args.rollout_steps, args.players_per_team), dtype=np.float32)
        values_buffer = np.zeros((args.rollout_steps, args.players_per_team), dtype=np.float32)
        rewards_buffer = np.zeros((args.rollout_steps, args.players_per_team), dtype=np.float32)
        dones = np.zeros(args.rollout_steps, dtype=np.float32)
        episode_starts = np.zeros((args.rollout_steps, 1), dtype=np.float32)
        infos: list[dict[str, object]] = []
        opponent_kinds: list[str] = []

        for step in range(args.rollout_steps):
            episode_starts[step, 0] = float(episode_start)
            if episode_start:
                hidden.zero_()
                episode_start = False
            observation_tensor = torch.from_numpy(left_view.observations).to(device)
            state_tensor = torch.from_numpy(left_view.central_state).to(device).unsqueeze(0)
            with torch.no_grad():
                means, log_stds, hidden = model.actor_step(observation_tensor, hidden)
                distribution = Normal(means, log_stds.exp())
                raw = distribution.rsample()
                actions = torch.tanh(raw)
                step_log_probs = squashed_log_prob(distribution, raw, actions)
                step_values = model.values(state_tensor, args.players_per_team).squeeze(0)
            right_actions = opponent.actions(env, right_view)
            next_left, next_right, rewards, done, info = env.step(
                actions.cpu().numpy(), right_actions
            )
            observations[step] = left_view.observations
            central_states[step] = left_view.central_state
            actions_buffer[step] = actions.cpu().numpy()
            log_probs[step] = step_log_probs.cpu().numpy()
            values_buffer[step] = step_values.cpu().numpy()
            rewards_buffer[step] = rewards
            dones[step] = float(done)
            infos.append(info)
            opponent_kinds.append(opponent.member.kind)
            if done:
                left_view, right_view, _ = env.reset()
                hidden = model.initial_hidden(args.players_per_team, device=device)
                episode_start = True
                opponent = OpponentController(pool.sample(), args.players_per_team, device=args.device)
            else:
                left_view, right_view = next_left, next_right

        with torch.no_grad():
            last_state = torch.from_numpy(left_view.central_state).to(device).unsqueeze(0)
            last_values = model.values(last_state, args.players_per_team).squeeze(0).cpu().numpy()
        advantages, returns = compute_gae(
            rewards_buffer,
            values_buffer,
            dones,
            last_values,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        advantage_tensor = torch.from_numpy(advantages).to(device)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
            advantage_tensor.std() + 1.0e-8
        )
        return_tensor = torch.from_numpy(returns).to(device)
        observation_tensor = torch.from_numpy(observations).to(device)
        state_tensor = torch.from_numpy(central_states).to(device)
        action_tensor = torch.from_numpy(actions_buffer).to(device)
        old_log_prob_tensor = torch.from_numpy(log_probs).to(device)
        start_tensor = torch.from_numpy(episode_starts).to(device)
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []

        for _ in range(args.epochs):
            means, log_stds, _ = model.actor_unroll(
                observation_tensor,
                initial_hidden,
                start_tensor,
            )
            distribution = Normal(means, log_stds.exp())
            safe_actions = torch.clamp(action_tensor, -0.999999, 0.999999)
            raw_actions = torch.atanh(safe_actions)
            new_log_probs = squashed_log_prob(distribution, raw_actions, safe_actions)
            ratios = torch.exp(new_log_probs - old_log_prob_tensor)
            unclipped = ratios * advantage_tensor
            clipped = torch.clamp(
                ratios,
                1.0 - args.clip_range,
                1.0 + args.clip_range,
            ) * advantage_tensor
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            predicted_values = model.values(state_tensor, args.players_per_team)
            value_loss = torch.square(predicted_values - return_tensor).mean()
            entropy = distribution.entropy().sum(dim=-1).mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))

        checkpoint = next_numbered_path(
            EXPERIMENT_ROOT / "checkpoints", "v043_mappo_local_v2", ".pt"
        )
        save_checkpoint(
            checkpoint,
            model,
            update=update,
            players_per_team=args.players_per_team,
            optimizer=optimizer,
            evidence_status="short_training_checkpoint_not_promoted",
        )
        snapshot_path = None
        if update % args.snapshot_interval == 0:
            snapshot_path = pool.add(model, update=update, players_per_team=args.players_per_team)
        metric_path = next_numbered_path(
            EXPERIMENT_ROOT / "metrics", "v043_mappo_local_v2_training", ".json"
        )
        safety_passed = all(
            bool(info.get("all_targets_sent")) and int(info.get("fallen_players", 1)) == 0
            for info in infos
        )
        payload = {
            "version": "v043_team_mappo",
            "observation_schema": "v2_decentralized_local_roles_and_own_touch_only",
            "method": "parameter_shared_gru_actor_centralized_critic_mappo",
            "evidence_status": "short_training_execution_only_not_match_competitive_not_promoted",
            "update": update,
            "players_per_team": args.players_per_team,
            "environment_steps_this_update": args.rollout_steps,
            "player_samples_this_update": args.rollout_steps * args.players_per_team,
            "mean_reward": float(rewards_buffer.mean()),
            "mean_ball_progress_m": float(np.mean([float(info["ball_progress_m"]) for info in infos])),
            "mean_visible_entity_ratio": float(
                np.mean([float(info["visible_entity_ratio"]) for info in infos])
            ),
            "mean_fallen_players": float(np.mean([float(info["fallen_players"]) for info in infos])),
            "turnover_steps": int(sum(bool(info["turnover"]) for info in infos)),
            "teammate_touch_change_steps": int(
                sum(bool(info["teammate_touch_change"]) for info in infos)
            ),
            "safety_passed": safety_passed,
            "mean_loss": float(np.mean(losses)),
            "mean_policy_loss": float(np.mean(policy_losses)),
            "mean_value_loss": float(np.mean(value_losses)),
            "opponent_kinds": sorted(set(opponent_kinds)),
            "opponent_pool_size": len(pool.snapshots),
            "snapshot_added": None if snapshot_path is None else str(snapshot_path),
            "checkpoint": str(checkpoint),
            "quality_passed": False,
            "quality_reason": "No held-out match league evaluation has been run; short PPO training cannot satisfy promotion gates.",
        }
        metric_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**payload, "metrics": str(metric_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
