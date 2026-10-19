"""Train one non-overwriting PPO chunk for v034 and immediately run fixed-seed evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for path in (PROJECT_ROOT, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from dynamic_kick_env import DynamicSingleLegKickEnv  # noqa: E402
from kick_training import (  # noqa: E402
    DEFAULT_CONFIG,
    latest_checkpoint,
    load_config,
    next_checkpoint_path,
    run_fixed_evaluation,
    unique_metrics_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    ppo_cfg = config["ppo"]
    timesteps = int(args.timesteps or config["training"]["short_chunk_timesteps"])
    if timesteps < int(ppo_cfg["n_steps"]):
        raise ValueError("timesteps must be at least one PPO rollout")

    checkpoint = next_checkpoint_path()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    resume_from = latest_checkpoint()
    env = Monitor(DynamicSingleLegKickEnv.from_config(config, seed=int(ppo_cfg["seed"])))
    if resume_from is not None:
        model = PPO.load(str(resume_from), env=env, device="cpu")
        if "max_log_std" in ppo_cfg:
            model.policy.log_std.data.clamp_(max=float(ppo_cfg["max_log_std"]))
        source = str(resume_from)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=float(ppo_cfg["learning_rate"]),
            n_steps=int(ppo_cfg["n_steps"]),
            batch_size=int(ppo_cfg["batch_size"]),
            n_epochs=int(ppo_cfg["n_epochs"]),
            gamma=float(ppo_cfg["gamma"]),
            gae_lambda=float(ppo_cfg["gae_lambda"]),
            clip_range=float(ppo_cfg["clip_range"]),
            ent_coef=float(ppo_cfg["ent_coef"]),
            policy_kwargs={
                "net_arch": list(ppo_cfg["policy_layers"]),
                "log_std_init": float(ppo_cfg.get("log_std_init", 0.0)),
            },
            seed=int(ppo_cfg["seed"]),
            device="cpu",
            verbose=1,
        )
        source = "fresh PPO residual initialized on v032/v033 behavior priors"

    started = time.time()
    try:
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)
        model.save(str(checkpoint))
    finally:
        env.close()
    elapsed = time.time() - started

    evaluation = run_fixed_evaluation(model, config)
    payload = {
        "checkpoint": str(checkpoint),
        "resumed_from": source,
        "config": str(args.config),
        "chunk_timesteps": timesteps,
        "total_model_timesteps": int(model.num_timesteps),
        "elapsed_seconds": elapsed,
        **evaluation,
    }
    output = unique_metrics_path(f"{checkpoint.stem}_training")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
