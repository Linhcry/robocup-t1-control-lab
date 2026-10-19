"""从 v037 检查点继续训练 v038 的一个高速控球课程阶段。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from experiments.v038_high_speed_dribble.curriculum import (  # noqa: E402
    DEFAULT_CONFIG,
    evaluate_stage,
    latest_stage_checkpoint,
    load_config,
    load_or_continue_model,
    next_stage_checkpoint,
    unique_metrics_path,
)
from experiments.v038_high_speed_dribble.high_speed_dribble_env import (  # noqa: E402
    HighSpeedResidualDribbleEnv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument(
        "--safe-reset-head",
        action="store_true",
        help="从 v037 重新加载特征层，但把有害残差输出头归零后再训练",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    timesteps = int(args.timesteps or config["ppo"]["default_chunk_timesteps"])
    training_seed = int(config["ppo"]["seed"]) + 100 * int(args.stage)
    env = Monitor(HighSpeedResidualDribbleEnv(config, args.stage, seed=training_seed))
    if args.safe_reset_head:
        if args.stage == 0:
            source_checkpoint = Path(config["bootstrap_model"])
            if not source_checkpoint.is_absolute():
                source_checkpoint = PROJECT_ROOT / source_checkpoint
        else:
            previous = latest_stage_checkpoint(args.stage - 1)
            if previous is None:
                raise FileNotFoundError(f"stage {args.stage - 1} checkpoint is missing")
            source_checkpoint = previous
        model = PPO.load(str(source_checkpoint), env=env, device="cpu")
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
        source = f"{source_checkpoint} + zeroed residual action head"
    else:
        model, source = load_or_continue_model(config, args.stage, env)
    if timesteps < int(model.n_steps):
        raise ValueError(f"timesteps must be at least one rollout ({model.n_steps})")
    model.learning_rate = float(config["ppo"]["learning_rate"])
    model.lr_schedule = lambda _: float(config["ppo"]["learning_rate"])
    model.policy.log_std.data.clamp_(max=float(config["ppo"]["max_log_std"]))
    checkpoint = next_stage_checkpoint(args.stage)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)
        model.policy.log_std.data.clamp_(max=float(config["ppo"]["max_log_std"]))
        model.save(str(checkpoint))
    finally:
        env.close()

    validation = evaluate_stage(model, config, args.stage)
    payload = {
        "checkpoint": str(checkpoint),
        "resumed_from": source,
        "config": str(args.config),
        "training_seed": training_seed,
        "safe_reset_action_head": bool(args.safe_reset_head),
        "chunk_timesteps": timesteps,
        "total_model_timesteps": int(model.num_timesteps),
        "elapsed_seconds": time.time() - started,
        **validation,
    }
    metrics = unique_metrics_path(f"{checkpoint.stem}_training")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(metrics)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
