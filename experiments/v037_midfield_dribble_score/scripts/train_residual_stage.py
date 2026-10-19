"""训练 v037-r2 的一个教师残差阶段。"""

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

from stable_baselines3.common.monitor import Monitor  # noqa: E402

from experiments.v037_midfield_dribble_score.curriculum import DEFAULT_CONFIG, load_config, unique_metrics_path  # noqa: E402
from experiments.v037_midfield_dribble_score.midfield_dribble_env import ResidualMidfieldDribbleScoreEnv  # noqa: E402
from experiments.v037_midfield_dribble_score.residual_curriculum import (  # noqa: E402
    evaluate_residual_stage,
    load_or_build_stage_model,
    next_residual_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    timesteps = int(args.timesteps or config["ppo"]["default_chunk_timesteps"])
    env = Monitor(ResidualMidfieldDribbleScoreEnv(config, args.stage, seed=38800 + args.stage))
    model, source = load_or_build_stage_model(config, args.stage, env)
    if timesteps < int(model.n_steps):
        raise ValueError(f"timesteps must be at least one rollout ({model.n_steps})")
    model.policy.log_std.data.clamp_(max=float(config["ppo"]["max_log_std"]))
    checkpoint = next_residual_checkpoint(args.stage)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)
        model.policy.log_std.data.clamp_(max=float(config["ppo"]["max_log_std"]))
        model.save(str(checkpoint))
    finally:
        env.close()
    payload = {
        "checkpoint": str(checkpoint),
        "resumed_from": source,
        "config": str(args.config),
        "chunk_timesteps": timesteps,
        "total_model_timesteps": int(model.num_timesteps),
        "elapsed_seconds": time.time() - started,
        **evaluate_residual_stage(model, config, args.stage),
    }
    metrics = unique_metrics_path(f"{checkpoint.stem}_training")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(metrics)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
