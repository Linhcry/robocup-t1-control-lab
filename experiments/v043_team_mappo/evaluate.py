"""Fixed-seed neural-team evaluation against the privileged role heuristic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.v043_team_mappo.artifacts import next_numbered_path
from experiments.v043_team_mappo.ball_events import (
    POSSESSION_EVIDENCE_SCHEMA,
    POSSESSION_HOLD_WINDOW_S,
)
from experiments.v043_team_mappo.environment import TeamSelfPlayEnv
from experiments.v043_team_mappo.networks import load_checkpoint
from experiments.v043_team_mappo.tactics import privileged_role_heuristic_actions
from experiments.v043_team_mappo.train_mappo import DEFAULT_LOW_LEVEL_MODEL


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--players-per-team", type=int, choices=range(3, 8), default=3)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-high-level-steps", type=int, default=300)
    parser.add_argument("--prepare-seconds", type=float, default=0.5)
    parser.add_argument("--low-level-model", type=Path, default=DEFAULT_LOW_LEVEL_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed-base", type=int, default=43100)
    parser.add_argument("--held-out", action="store_true")
    args = parser.parse_args()

    model, checkpoint_payload = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    trained_players = int(checkpoint_payload.get("players_per_team", -1))
    if trained_players != args.players_per_team:
        raise ValueError(
            f"checkpoint was trained with {trained_players} players, requested {args.players_per_team}"
        )
    env = TeamSelfPlayEnv(
        args.low_level_model,
        players_per_team=args.players_per_team,
        episode_high_level_steps=args.episode_high_level_steps,
        prepare_seconds=args.prepare_seconds,
        seed=args.seed_base,
    )
    episodes: list[dict[str, object]] = []
    device = torch.device(args.device)
    with torch.inference_mode():
        for episode_index in range(args.episodes):
            seed = args.seed_base + episode_index
            env.rng = np.random.default_rng(seed)
            left, right, _ = env.reset()
            hidden = model.initial_hidden(args.players_per_team, device=device)
            possession_steps = 0
            opponent_possession_steps = 0
            neutral_possession_steps = 0
            active_contact_possession_steps = 0
            recent_touch_hold_possession_steps = 0
            learner_touch_events = 0
            opponent_touch_events = 0
            teammate_touch_changes = 0
            turnover_steps = 0
            progress = 0.0
            maximum_fallen = 0
            final_info: dict[str, object] = {}
            for step in range(args.episode_high_level_steps):
                actions, hidden = model.deterministic_step(
                    torch.from_numpy(left.observations).to(device), hidden
                )
                opponent_actions = privileged_role_heuristic_actions(
                    env.core.simulation,
                    env.core.right_players,
                    env.core.right_teacher_roles,
                    side_sign=-1,
                )
                left, right, _, done, info = env.step(
                    actions.cpu().numpy(), opponent_actions
                )
                final_info = info
                progress += float(info["ball_progress_m"])
                possession_side = int(info["possession_side"])
                possession_source = str(info["possession_evidence_source"])
                if possession_source not in {"active_contact", "recent_touch_hold", "neutral"}:
                    raise RuntimeError(f"unknown possession evidence source: {possession_source}")
                if (possession_side == 0) != (possession_source == "neutral"):
                    raise RuntimeError("possession side is not consistent with real-contact evidence")
                possession_steps += int(possession_side == 1)
                opponent_possession_steps += int(possession_side == -1)
                neutral_possession_steps += int(possession_side == 0)
                active_contact_possession_steps += int(possession_source == "active_contact")
                recent_touch_hold_possession_steps += int(possession_source == "recent_touch_hold")
                learner_touch_events += int(info["learner_touch_events"])
                opponent_touch_events += int(info["opponent_touch_events"])
                teammate_touch_changes += int(bool(info["teammate_touch_change"]))
                turnover_steps += int(bool(info["turnover"]))
                maximum_fallen = max(maximum_fallen, int(info["fallen_players"]))
                if done:
                    break
            total_possession = possession_steps + opponent_possession_steps
            episodes.append(
                {
                    "seed": seed,
                    "high_level_steps": step + 1,
                    "left_score": int(final_info.get("left_score", 0)),
                    "right_score": int(final_info.get("right_score", 0)),
                    "ball_progress_m": progress,
                    "learner_possession_ratio": (
                        0.0 if total_possession == 0 else possession_steps / total_possession
                    ),
                    "possession_evidence_schema": POSSESSION_EVIDENCE_SCHEMA,
                    "possession_hold_window_s": POSSESSION_HOLD_WINDOW_S,
                    "possession_sample_steps": step + 1,
                    "learner_contact_possession_steps": possession_steps,
                    "opponent_contact_possession_steps": opponent_possession_steps,
                    "neutral_possession_steps": neutral_possession_steps,
                    "active_contact_possession_steps": active_contact_possession_steps,
                    "recent_touch_hold_possession_steps": recent_touch_hold_possession_steps,
                    "learner_touch_event_count": learner_touch_events,
                    "opponent_touch_event_count": opponent_touch_events,
                    "teammate_touch_change_proxy_count": teammate_touch_changes,
                    "turnover_step_count": turnover_steps,
                    "maximum_fallen_players": maximum_fallen,
                    "all_targets_sent": bool(final_info.get("all_targets_sent", False)),
                    "simulated_seconds": float(
                        (step + 1) * env.action_repeat * env.core.simulation.timestep
                    ),
                    "true_pass_attempts": 0,
                    "true_pass_completions": 0,
                    "opponent_id": "privileged_dynamic_role_heuristic",
                }
            )

    wins = sum(int(row["left_score"]) > int(row["right_score"]) for row in episodes)
    draws = sum(int(row["left_score"]) == int(row["right_score"]) for row in episodes)
    losses = len(episodes) - wins - draws
    total_seconds = sum(float(row["simulated_seconds"]) for row in episodes)
    total_learner_possession = sum(
        int(row["learner_contact_possession_steps"]) for row in episodes
    )
    total_opponent_possession = sum(
        int(row["opponent_contact_possession_steps"]) for row in episodes
    )
    total_possession_samples = sum(int(row["possession_sample_steps"]) for row in episodes)
    total_contact_backed_possession = total_learner_possession + total_opponent_possession
    summary = {
        "episodes": len(episodes),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / max(1, len(episodes)),
        "mean_ball_progress_m": float(np.mean([float(row["ball_progress_m"]) for row in episodes])),
        "mean_learner_possession_ratio": float(
            0.0
            if total_contact_backed_possession == 0
            else total_learner_possession / total_contact_backed_possession
        ),
        "contact_backed_possession_sample_rate": float(
            total_contact_backed_possession / max(1, total_possession_samples)
        ),
        "learner_touch_event_count": int(
            sum(int(row["learner_touch_event_count"]) for row in episodes)
        ),
        "opponent_touch_event_count": int(
            sum(int(row["opponent_touch_event_count"]) for row in episodes)
        ),
        "fall_episode_rate": float(
            np.mean([int(row["maximum_fallen_players"]) > 0 for row in episodes])
        ),
        "all_23_targets_sent_rate": float(
            np.mean([bool(row["all_targets_sent"]) for row in episodes])
        ),
        "teammate_touch_change_proxy_total": int(
            sum(int(row["teammate_touch_change_proxy_count"]) for row in episodes)
        ),
        "true_pass_attempts": 0,
        "true_pass_completions": 0,
        "true_pass_completion_rate": 0.0,
        "mean_goals_conceded_per_10min": float(
            600.0 * sum(int(row["right_score"]) for row in episodes) / total_seconds
        ),
        "minimum_match_duration_s": float(
            min(float(row["simulated_seconds"]) for row in episodes)
        ),
        "distinct_opponents": 1,
    }
    output = next_numbered_path(EXPERIMENT_ROOT / "metrics", "v043_fixed_eval", ".json")
    payload = {
        "version": "v043_team_mappo",
        "observation_schema": "v2_decentralized_local_roles_and_own_touch_only",
        "possession_evidence_schema": POSSESSION_EVIDENCE_SCHEMA,
        "possession_hold_window_s": POSSESSION_HOLD_WINDOW_S,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.resolve().read_bytes()).hexdigest(),
        "checkpoint_format": checkpoint_payload.get("format"),
        "players_per_team": args.players_per_team,
        "checkpoint_evidence_status": checkpoint_payload.get("evidence_status"),
        "learner_policy": "deterministic_neural_gru_actor",
        "opponent_policy": "privileged_dynamic_role_heuristic",
        "evaluation_seeds": [args.seed_base + index for index in range(args.episodes)],
        "evaluation_split": "held_out" if args.held_out else "fixed_validation",
        "opponent_pool_members": ["privileged_dynamic_role_heuristic"],
        "validated_team_skill_interface": False,
        "summary": summary,
        "episodes": episodes,
        "quality_passed": False,
        "quality_reason": (
            "The current action interface has no validated pass/dive/get-up skill and this run does not meet "
            "the configured 20-match promotion protocol. Teammate contact changes are only a proxy, not pass proof."
        ),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
