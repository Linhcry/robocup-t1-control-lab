"""训练一个 v037 距离阶段；未通过前一阶段时拒绝越级。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from experiments.v037_midfield_dribble_score.curriculum import (  # noqa: E402
    DEFAULT_CONFIG,
    evaluate_stage,
    load_config,
    next_stage_checkpoint,
    source_checkpoint,
    unique_metrics_path,
)
from experiments.v037_midfield_dribble_score.midfield_dribble_env import MidfieldDribbleScoreEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    stage_index = int(args.stage)
    source = source_checkpoint(config, stage_index)
    timesteps = int(args.timesteps or config["ppo"]["default_chunk_timesteps"])
    env = Monitor(MidfieldDribbleScoreEnv(config, stage_index, seed=38000 + stage_index))
    model = PPO.load(str(source), env=env, device="cpu")
    if timesteps < int(model.n_steps):
        raise ValueError(f"timesteps must be at least one PPO rollout ({model.n_steps})")
    model.policy.log_std.data.clamp_(max=float(config["ppo"]["max_log_std"]))
    checkpoint = next_stage_checkpoint(stage_index)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    try:
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)
        model.policy.log_std.data.clamp_(max=float(config["ppo"]["max_log_std"]))
        model.save(str(checkpoint))
    finally:
        env.close()
    evaluation = evaluate_stage(model, config, stage_index)
    payload = {
        "checkpoint": str(checkpoint),
        "resumed_from": str(source),
        "config": str(args.config),
        "chunk_timesteps": timesteps,
        "total_model_timesteps": int(model.num_timesteps),
        "elapsed_seconds": time.time() - started,
        **evaluation,
    }
    metrics = unique_metrics_path(f"{checkpoint.stem}_training")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(metrics)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
