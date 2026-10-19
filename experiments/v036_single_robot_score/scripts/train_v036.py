"""行为克隆热启动后继续一段 PPO，并立即执行固定种子射门评估。"""

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

from experiments.v036_single_robot_score.single_robot_score_env import SingleRobotScoreEnv  # noqa: E402
from experiments.v036_single_robot_score.training import (  # noqa: E402
    DEFAULT_CONFIG,
    behavior_clone_actor,
    build_ppo,
    collect_teacher_demonstrations,
    evaluate_policy,
    latest_checkpoint,
    load_config,
    next_checkpoint_path,
    unique_metrics_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    ppo_cfg = config["ppo"]
    bc_cfg = config["behavior_cloning"]
    timesteps = int(args.timesteps or config["training"]["short_chunk_timesteps"])
    if timesteps < int(ppo_cfg["n_steps"]):
        raise ValueError("timesteps must be at least one complete PPO rollout")

    env = Monitor(SingleRobotScoreEnv(config, seed=int(ppo_cfg["seed"])))
    resume_from = latest_checkpoint()
    bc_payload = None
    if resume_from is None:
        model = build_ppo(config, env)
        observations, actions, demo_summary = collect_teacher_demonstrations(
            config,
            sample_count=int(bc_cfg["samples"]),
            seed=int(ppo_cfg["seed"]) + 1,
        )
        bc_losses = behavior_clone_actor(
            model,
            observations,
            actions,
            epochs=int(bc_cfg["epochs"]),
            batch_size=int(bc_cfg["batch_size"]),
            learning_rate=float(bc_cfg["learning_rate"]),
            seed=int(ppo_cfg["seed"]),
        )
        warm_start = next_checkpoint_path()
        warm_start.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(warm_start))
        bc_evaluation = evaluate_policy(model, config)
        bc_payload = {
            "checkpoint": str(warm_start),
            "method": "behavior_cloning_from_v033_teacher",
            "demonstrations": demo_summary,
            "first_epoch_mse": bc_losses[0],
            "final_epoch_mse": bc_losses[-1],
            **bc_evaluation,
        }
        bc_metrics = unique_metrics_path(f"{warm_start.stem}_behavior_cloning")
        bc_metrics.write_text(json.dumps(bc_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        source = str(warm_start)
    else:
        model = PPO.load(str(resume_from), env=env, device="cpu")
        source = str(resume_from)

    model.policy.log_std.data.clamp_(max=float(ppo_cfg["max_log_std"]))
    checkpoint = next_checkpoint_path()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)
        model.policy.log_std.data.clamp_(max=float(ppo_cfg["max_log_std"]))
        model.save(str(checkpoint))
    finally:
        env.close()
    elapsed = time.time() - started

    evaluation = evaluate_policy(model, config)
    payload = {
        "checkpoint": str(checkpoint),
        "resumed_from": source,
        "config": str(args.config),
        "chunk_timesteps": timesteps,
        "total_model_timesteps": int(model.num_timesteps),
        "elapsed_seconds": elapsed,
        "behavior_cloning_created_this_run": bc_payload is not None,
        **evaluation,
    }
    output = unique_metrics_path(f"{checkpoint.stem}_training")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
