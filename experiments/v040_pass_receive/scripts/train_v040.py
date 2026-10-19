from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from experiments.v040_pass_receive.pass_receive_env import PassReceiveEnv
from experiments.v040_pass_receive.runner import load_json, next_iteration_path, validate_model_spaces, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage", type=int, default=0)
    args = parser.parse_args()
    config = load_json(ROOT / "configs" / "v040_pass_receive.json")
    ppo = config["ppo"]
    env = PassReceiveEnv(config, seed=int(ppo["seed"]))
    if args.stage != 0:
        raise SystemExit("v040 只有 stage 0 已实现；3-12米阶段必须先通过动态踢球门槛")
    if args.resume is None:
        model = PPO(
            "MlpPolicy", env, learning_rate=float(ppo["learning_rate"]), n_steps=int(ppo["n_steps"]),
            batch_size=int(ppo["batch_size"]), n_epochs=int(ppo["n_epochs"]), gamma=float(ppo["gamma"]),
            gae_lambda=float(ppo["gae_lambda"]), clip_range=float(ppo["clip_range"]),
            ent_coef=float(ppo["ent_coef"]), policy_kwargs={"net_arch": list(ppo["policy_layers"])},
            seed=int(ppo["seed"]), verbose=1, device="cpu",
        )
        model.policy.action_net.weight.data.zero_()
        model.policy.action_net.bias.data.zero_()
        model.policy.log_std.data.fill_(-2.0)
        reset_num_timesteps = True
    else:
        model = PPO.load(args.resume, env=env, device="cpu")
        validate_model_spaces(model, env)
        reset_num_timesteps = False
    model.learn(total_timesteps=int(args.timesteps), reset_num_timesteps=reset_num_timesteps)
    iteration, checkpoint = next_iteration_path(ROOT / "checkpoints", "v040_policy", ".zip")
    model.save(checkpoint.with_suffix(""))
    record = ROOT / "metrics" / f"v040_policy_iter_{iteration:04d}_training.json"
    write_json(
        record,
        {"version": config["version"], "stage": int(args.stage), "chunk_timesteps": int(args.timesteps),
         "total_model_timesteps": int(model.num_timesteps), "resumed_from": None if args.resume is None else str(args.resume),
         "checkpoint": str(checkpoint)},
    )
    env.close()
    print(checkpoint)


if __name__ == "__main__":
    main()
