"""Vectorized CPU smoke; it cannot promote tackle quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v048_fullbody_legal_tackle.rcss_mjlab_task import (  # noqa: E402
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    exact_rcss_tackle_env_cfg,
)
from experiments.v048_fullbody_legal_tackle.scripts._evidence import write_new_json  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=48000)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    if args.steps <= 0 or args.num_envs <= 0:
        parser.error("steps and num-envs must be positive")
    return args


def main() -> int:
    args = _args()
    from mjlab.envs import ManagerBasedRlEnv

    cfg = exact_rcss_tackle_env_cfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg, device="cpu")
    maximum_reward = float("-inf")
    minimum_reward = float("inf")
    terminated_count = 0
    timeout_count = 0
    finite = True
    try:
        observations, _ = env.reset(seed=args.seed)
        actor = observations["actor"]
        critic = observations["critic"]
        finite &= bool(torch.isfinite(actor).all() and torch.isfinite(critic).all())
        for step in range(args.steps):
            # Small deterministic excitation checks all 23 action routes without
            # claiming any learned tackle behavior.
            action = torch.full(
                (args.num_envs, ACTION_DIM),
                0.02 * ((step % 3) - 1),
                device="cpu",
            )
            observations, reward, terminated, timeout, _ = env.step(action)
            actor = observations["actor"]
            critic = observations["critic"]
            finite &= bool(
                torch.isfinite(actor).all()
                and torch.isfinite(critic).all()
                and torch.isfinite(reward).all()
                and torch.isfinite(env.sim.data.qpos).all()
                and torch.isfinite(env.sim.data.qvel).all()
                and torch.isfinite(env.sim.data.ctrl).all()
            )
            maximum_reward = max(maximum_reward, float(reward.max()))
            minimum_reward = min(minimum_reward, float(reward.min()))
            terminated_count += int(terminated.sum())
            timeout_count += int(timeout.sum())
        action_term = env.action_manager.get_term("defender_fullbody")
        payload = {
            "version": "v048",
            "kind": "cpu_smoke_8_steps",
            "device": "cpu",
            "seed": args.seed,
            "steps": args.steps,
            "num_envs": env.num_envs,
            "actor_observation_dim": int(actor.shape[-1]),
            "critic_observation_dim": int(critic.shape[-1]),
            "action_dim": int(env.action_manager.total_action_dim),
            "action_target_names": list(action_term.target_names),
            "finite": finite,
            "minimum_reward": minimum_reward,
            "maximum_reward": maximum_reward,
            "terminated_count": terminated_count,
            "timeout_count": timeout_count,
            "smoke_passed": bool(
                finite
                and actor.shape[-1] == ACTOR_OBSERVATION_DIM
                and env.action_manager.total_action_dim == ACTION_DIM
            ),
            "quality_passed": False,
            "note": "Smoke proves construction and finite stepping only; no checkpoint was trained or promoted."
        }
    finally:
        env.close()
    path = write_new_json("v048_cpu_smoke", payload)
    print(json.dumps({"evidence": str(path), **payload}, ensure_ascii=False, indent=2))
    return 0 if payload["smoke_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
