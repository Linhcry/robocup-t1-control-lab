"""Vectorized source-domain shortlist evaluation for v044 checkpoints.

This script intentionally never promotes a checkpoint.  It compares candidates
inside the official Booster T1 MJLab model using identical random falls so that a
reasonable network can initialize v045.  Competition quality remains an RCSS-only
decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


TASK_ID = "Mjlab-Getup-Flat-Booster-T1"
FORMAL_SEEDS = (44700, 44701, 44702)
CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--steps", type=int, default=299)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or not 1 <= args.steps < 300:
        parser.error("num-envs must be positive and steps must be in [1, 299]")
    if bool(args.checkpoint) == bool(args.run_dir):
        parser.error("provide exactly one of --checkpoint or --run-dir")
    args.seed = list(args.seed) if args.seed else list(FORMAL_SEEDS)
    if len(set(args.seed)) != len(args.seed):
        parser.error("seed list contains duplicates")
    if not args.engineering_smoke and (
        tuple(args.seed) != FORMAL_SEEDS
        or args.num_envs != 256
        or args.steps != 299
    ):
        parser.error(
            "the source-shortlist protocol requires seeds 44700/44701/44702, "
            "256 environments and 299 steps; use --engineering-smoke to override"
        )
    return args


def _checkpoints_from_args(args: argparse.Namespace) -> list[Path]:
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        indexed: list[tuple[int, Path]] = []
        for path in run_dir.glob("model_*.pt"):
            match = CHECKPOINT_RE.match(path.name)
            if match:
                indexed.append((int(match.group(1)), path.resolve()))
        indexed.sort(key=lambda item: item[0])
        checkpoints = [path for _, path in indexed]
    else:
        checkpoints = [path.resolve() for path in args.checkpoint]
    if not checkpoints:
        raise ValueError("no model_N.pt checkpoints were selected")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if len({str(path) for path in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint list contains duplicates")
    return checkpoints


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical evidence: {output}")
    checkpoints = _checkpoints_from_args(args)

    # Importing this module registers the official task without altering it.
    import mjlab_playground.getup.config.t1  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    device = args.device
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.terrain.num_envs = args.num_envs
    env_cfg.seed = args.seed[0]
    # The play configuration already forces every reset to be fallen.
    fall_probability = float(
        env_cfg.events["reset_fallen_or_standing"].params["fall_probability"]
    )
    if fall_probability != 1.0:
        raise ValueError("source candidate evaluation requires 100% fallen resets")

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    results: list[dict[str, Any]] = []
    try:
        for checkpoint in checkpoints:
            runner.load(
                str(checkpoint),
                load_cfg={"actor": True},
                strict=True,
                map_location=device,
            )
            policy = runner.get_inference_policy(device=device)
            sustained_steps = round(3.0 / float(env.unwrapped.step_dt))
            per_seed: list[dict[str, Any]] = []
            for seed in args.seed:
                env.seed(seed)
                observations, _ = env.reset()
                asset = env.unwrapped.scene["robot"]
                stood_once = torch.zeros(
                    args.num_envs, dtype=torch.bool, device=device
                )
                consecutive = torch.zeros(
                    args.num_envs, dtype=torch.int64, device=device
                )
                max_consecutive = torch.zeros_like(consecutive)
                early_done = torch.zeros(
                    args.num_envs, dtype=torch.bool, device=device
                )
                # MJLab updates contact-sensor buffers during env.step().  Using
                # inference_mode would turn those buffers into inference tensors,
                # which cannot be reset in-place for the next independent seed.
                # no_grad still disables policy autograd without altering buffer
                # tensor semantics.
                with torch.no_grad():
                    for step in range(args.steps):
                        actions = policy(observations)
                        observations, _, dones, _ = env.step(actions)
                        done_now = dones.to(dtype=torch.bool)
                        gravity = asset.data.projected_gravity_b
                        upright_error = torch.sum(
                            torch.square(
                                gravity
                                - torch.tensor(
                                    [0.0, 0.0, -1.0],
                                    device=device,
                                    dtype=gravity.dtype,
                                )
                            ),
                            dim=-1,
                        )
                        height = asset.data.root_link_pos_w[:, 2]
                        standing = (upright_error < 0.05) & ((0.67 - height) < 0.02)
                        # VecEnv auto-resets completed worlds.  A post-reset
                        # standing state must never become a second attempt for
                        # the original fallen sample.  Any early termination
                        # permanently invalidates that world's success metrics.
                        if step < args.steps - 1:
                            early_done |= done_now
                        standing &= ~done_now & ~early_done
                        stood_once |= standing
                        consecutive = torch.where(
                            standing, consecutive + 1, torch.zeros_like(consecutive)
                        )
                        max_consecutive = torch.maximum(
                            max_consecutive, consecutive
                        )
                stood_once &= ~early_done
                sustained = (max_consecutive >= sustained_steps) & ~early_done
                per_seed.append(
                    {
                        "seed": seed,
                        "stood_once_count": int(stood_once.sum().item()),
                        "stood_once_rate": float(stood_once.float().mean().item()),
                        "continuous_upright_3s_count": int(sustained.sum().item()),
                        "continuous_upright_3s_rate": float(
                            sustained.float().mean().item()
                        ),
                        "early_termination_count": int(early_done.sum().item()),
                        "early_termination_rate": float(
                            early_done.float().mean().item()
                        ),
                        "maximum_continuous_upright_steps": int(
                            max_consecutive.max().item()
                        ),
                    }
                )
            continuous_rates = [
                item["continuous_upright_3s_rate"] for item in per_seed
            ]
            stood_rates = [item["stood_once_rate"] for item in per_seed]
            early_rates = [item["early_termination_rate"] for item in per_seed]
            results.append(
                {
                    "checkpoint": {
                        "path": str(checkpoint),
                        "sha256": _sha256(checkpoint),
                    },
                    "per_seed": per_seed,
                    "minimum_continuous_upright_3s_rate": min(continuous_rates),
                    "mean_continuous_upright_3s_rate": sum(continuous_rates)
                    / len(continuous_rates),
                    "minimum_stood_once_rate": min(stood_rates),
                    "mean_stood_once_rate": sum(stood_rates) / len(stood_rates),
                    "mean_early_termination_rate": sum(early_rates) / len(early_rates),
                }
            )
    finally:
        env.close()

    ranking = sorted(
        range(len(results)),
        key=lambda index: (
            results[index]["minimum_continuous_upright_3s_rate"],
            results[index]["mean_continuous_upright_3s_rate"],
            results[index]["minimum_stood_once_rate"],
            results[index]["mean_stood_once_rate"],
            -results[index]["mean_early_termination_rate"],
        ),
        reverse=True,
    )
    formal_shortlist_protocol = bool(
        not args.engineering_smoke
        and tuple(args.seed) == FORMAL_SEEDS
        and args.num_envs == 256
        and args.steps == 299
    )
    payload = {
        "schema_version": "v044_source_candidate_eval_v2",
        "experiment": "v044_mjlab_t1_getup",
        "backend": TASK_ID,
        "device": device,
        "seeds": args.seed,
        "num_environments": args.num_envs,
        "fall_probability": fall_probability,
        "steps": args.steps,
        "duration_s": args.steps * float(env.unwrapped.step_dt),
        "identical_random_stream_replayed_for_each_checkpoint": True,
        "results": results,
        "ranking_by_source_metrics": ranking,
        "formal_source_shortlist_protocol": formal_shortlist_protocol,
        "selected_checkpoint_for_target_adaptation": (
            results[ranking[0]]["checkpoint"] if formal_shortlist_protocol else None
        ),
        "source_shortlist_only": True,
        "validation_passed": False,
        "quality_passed": False,
        "reason_quality_is_false": (
            "This evaluation uses the official source robot and checks only height "
            "and orientation. It cannot replace RCSS contact safety, four-pose "
            "recovery, locomotion handoff, fixed validation, or held-out testing."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
