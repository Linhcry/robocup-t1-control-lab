from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from experiments.v040_pass_receive.runner import load_json, next_iteration_path, validate_model_spaces, write_json
from experiments.v041_active_tackle.active_tackle_env import ActiveTackleEnv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage", type=int, default=0)
    args = parser.parse_args()
    config = load_json(ROOT / "configs" / "v041_active_tackle.json")
    ppo = config["ppo"]
    env = ActiveTackleEnv(config, seed=int(ppo["seed"]))
    if args.stage != 0:
        raise SystemExit("v041 只有 stage 0 直线攻击者已实现；多对手池阶段尚未开放")
    if args.resume is None:
        model = PPO(
            "MlpPolicy", env, learning_rate=float(ppo["learning_rate"]), n_steps=int(ppo["n_steps"]),
            batch_size=int(ppo["batch_size"]), n_epochs=int(ppo["n_epochs"]), gamma=float(ppo["gamma"]),
            gae_lambda=float(ppo["gae_lambda"]), clip_range=float(ppo["clip_range"]),
            ent_coef=float(ppo["ent_coef"]), policy_kwargs={"net_arch": list(ppo["policy_layers"])},
            seed=int(ppo["seed"]), verbose=1, device="cpu",
        )
        model.policy.action_net.weight.data.zero_(); model.policy.action_net.bias.data.zero_(); model.policy.log_std.data.fill_(-2.0)
        reset_num_timesteps = True
    else:
        model = PPO.load(args.resume, env=env, device="cpu")
        validate_model_spaces(model, env)
        reset_num_timesteps = False
    model.learn(total_timesteps=int(args.timesteps), reset_num_timesteps=reset_num_timesteps)
    iteration, checkpoint = next_iteration_path(ROOT / "checkpoints", "v041_policy", ".zip")
    model.save(checkpoint.with_suffix(""))
    record = ROOT / "metrics" / f"v041_policy_iter_{iteration:04d}_training.json"
    write_json(record, {"version": config["version"], "stage": int(args.stage), "chunk_timesteps": int(args.timesteps),
                       "total_model_timesteps": int(model.num_timesteps), "resumed_from": None if args.resume is None else str(args.resume),
                       "checkpoint": str(checkpoint)})
    env.close()
    print(checkpoint)


if __name__ == "__main__":
    main()
