#!/usr/bin/env python3
"""Run eight real CPU policy steps and write immutable, non-quality evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _unique_output() -> Path:
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = root / f"v047_cpu_smoke_8_steps_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v047 smoke filename remains")


def main() -> int:
    from mjlab.envs import ManagerBasedRlEnv

    from experiments.v047_fullbody_goalkeeper import mdp
    from experiments.v047_fullbody_goalkeeper.rcss_goalkeeper_task import (
        rcss_fullbody_goalkeeper_env_cfg,
    )

    cfg = rcss_fullbody_goalkeeper_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.scene.terrain.num_envs = 1
    cfg.events["reset_goalkeeper_scenario"].params["stage_override"] = 0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    observations, _ = env.reset(seed=47001)
    actor_shape = tuple(observations["actor"].shape)
    critic_shape = tuple(observations["critic"].shape)
    ball = env.scene["ball"]
    initial_ball_velocity = ball.data.root_link_lin_vel_w[0].detach().cpu().tolist()
    finite_steps = []
    rewards = []
    for _ in range(8):
        action = torch.zeros((1, mdp.ACTION_DIM), device=env.device)
        observations, reward, terminated, truncated, _ = env.step(action)
        finite = bool(
            torch.isfinite(observations["actor"]).all()
            and torch.isfinite(observations["critic"]).all()
            and torch.isfinite(reward).all()
        )
        finite_steps.append(finite)
        rewards.append(float(reward[0].item()))
        if bool(terminated[0] or truncated[0]):
            break
    state = mdp.goalkeeper_state(env)
    stage_after_rollout = int(state.curriculum_stage[0].item())
    nonfinite_after_rollout = bool(state.nonfinite[0].item())

    # Exercise the exact runtime mutation used by the fixed/held-out evaluator.
    # One reset from every requested cell proves that the 27-cell contract is
    # reachable without replacing the policy or editing the task source.
    reset_cfg = env.event_manager.get_term_cfg("reset_goalkeeper_scenario")
    reset_cfg.params["stage_override"] = 3
    lateral_edges = (-1.78, -0.61, 0.61, 1.78)
    speed_edges = (3.0, 5.0, 7.0, 9.0)
    height_edges = (mdp.BALL_RADIUS_M, 0.35, 0.75, 1.12)
    evaluation_cells = []
    for flat_index in range(27):
        cell = (
            flat_index % 3,
            (flat_index // 3) % 3,
            (flat_index // 9) % 3,
        )
        reset_cfg.params["evaluation_cell"] = cell
        cell_obs, _ = env.reset(seed=47100 + flat_index)
        state = mdp.goalkeeper_state(env)
        target_y = float(state.shot_target_y[0].item())
        forward_speed = float(state.shot_forward_speed[0].item())
        target_z = float(state.shot_target_z[0].item())
        li, si, hi = cell
        in_requested_cell = (
            lateral_edges[li] <= target_y <= lateral_edges[li + 1]
            and speed_edges[si] <= forward_speed <= speed_edges[si + 1]
            and height_edges[hi] <= target_z <= height_edges[hi + 1]
        )
        evaluation_cells.append(
            {
                "cell": list(cell),
                "shot_target_y_m": target_y,
                "shot_forward_speed_mps": forward_speed,
                "shot_target_z_m": target_z,
                "in_requested_cell": in_requested_cell,
                "observations_finite": bool(
                    torch.isfinite(cell_obs["actor"]).all()
                    and torch.isfinite(cell_obs["critic"]).all()
                ),
            }
        )

    reset_cfg.params["evaluation_cell"] = (2, 1, 0)
    env.reset(seed=47999)
    replay_a = torch.stack(
        (
            mdp.goalkeeper_state(env).shot_target_y[0],
            mdp.goalkeeper_state(env).shot_forward_speed[0],
            mdp.goalkeeper_state(env).shot_target_z[0],
        )
    ).clone()
    env.reset(seed=47999)
    replay_b = torch.stack(
        (
            mdp.goalkeeper_state(env).shot_target_y[0],
            mdp.goalkeeper_state(env).shot_forward_speed[0],
            mdp.goalkeeper_state(env).shot_target_z[0],
        )
    ).clone()
    checks = {
        "actor_shape_is_1x87": actor_shape == (1, mdp.ACTOR_OBS_DIM),
        "critic_shape_is_1x107": critic_shape == (1, mdp.CRITIC_OBS_DIM),
        "action_dim_is_23": env.action_manager.total_action_dim == 23,
        "all_eight_steps_executed": len(finite_steps) == 8,
        "all_steps_finite": len(finite_steps) == 8 and all(finite_steps),
        "shot_moves_toward_keeper": initial_ball_velocity[0] < -3.0,
        "curriculum_stage_is_zero": stage_after_rollout == 0,
        "no_nonfinite_state_recorded": not nonfinite_after_rollout,
        "all_27_evaluation_cells_exercised": len(evaluation_cells) == 27,
        "all_evaluation_resets_are_in_requested_cell": all(
            row["in_requested_cell"] for row in evaluation_cells
        ),
        "all_evaluation_reset_observations_are_finite": all(
            row["observations_finite"] for row in evaluation_cells
        ),
        "same_seed_and_cell_replay_exactly": bool(torch.equal(replay_a, replay_b)),
    }
    passed = bool(checks) and all(checks.values())
    output = _unique_output()
    payload = {
        "schema_version": "v047_cpu_smoke_8_steps_v1",
        "seed": 47001,
        "device": "cpu",
        "steps_executed": len(finite_steps),
        "actor_observation_shape": actor_shape,
        "critic_observation_shape": critic_shape,
        "action_dim": env.action_manager.total_action_dim,
        "initial_ball_velocity_mps": initial_ball_velocity,
        "rewards": rewards,
        "evaluation_cells": evaluation_cells,
        "deterministic_replay": {
            "seed": 47999,
            "cell": [2, 1, 0],
            "first": replay_a.detach().cpu().tolist(),
            "second": replay_b.detach().cpu().tolist(),
        },
        "checks": checks,
        "smoke_passed": passed,
        "gpu_parallel_probe_completed": False,
        "training_completed": False,
        "fixed_validation_completed": False,
        "held_out_test_completed": False,
        "deployment_ready": False,
        "quality_passed": False,
        "scope_note": "Eight CPU steps prove construction and tensor flow only; they do not train or validate goalkeeper ability."
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    env.close()
    print(
        json.dumps(
            {"output": str(output), "smoke_passed": passed, "quality_passed": False},
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
