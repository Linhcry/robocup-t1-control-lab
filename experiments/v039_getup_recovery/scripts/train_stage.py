"""Short or full PPO training entry point for one gated v039 stage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v039_getup_recovery.artifacts import next_available_path, write_json  # noqa: E402
from experiments.v039_getup_recovery.getup_env import T1GetUpEnv, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=39001)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--previous-stage-proof",
        type=Path,
        help="fixed-validation JSON proving that the resumed checkpoint passed stage N-1",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def validate_previous_stage_proof(
    *, stage: int, resume: Path | None, proof_path: Path | None
) -> dict[str, object] | None:
    """Require stage N to resume the exact checkpoint that passed stage N-1."""

    if stage <= 0:
        if proof_path is not None:
            raise ValueError("--previous-stage-proof is only valid for stages 1-3")
        return None
    if resume is None:
        raise ValueError(
            "Stages 1-3 require --resume from the checkpoint that passed the previous stage."
        )
    if proof_path is None:
        raise ValueError(
            "Stages 1-3 require --previous-stage-proof; merely supplying --resume is not a quality gate."
        )
    resume_resolved = resume.expanduser().resolve(strict=True)
    proof_resolved = proof_path.expanduser().resolve(strict=True)
    proof_bytes = proof_resolved.read_bytes()
    try:
        proof = json.loads(proof_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid previous-stage proof JSON: {proof_resolved}") from exc
    expected_stage = stage - 1
    if int(proof.get("stage", -1)) != expected_stage:
        raise ValueError(
            f"Previous-stage proof must be for stage {expected_stage}, got {proof.get('stage')}"
        )
    if proof.get("evaluation_split") != "fixed_validation":
        raise ValueError(
            "Previous-stage proof must be a fixed_validation artifact produced by evaluate_stage.py"
        )
    summary = proof.get("summary")
    if not isinstance(summary, dict) or summary.get("quality_passed") is not True:
        raise ValueError("Previous-stage proof has quality_passed != true")
    proof_checkpoint = proof.get("checkpoint")
    if not isinstance(proof_checkpoint, str) or not proof_checkpoint:
        raise ValueError("Previous-stage proof does not identify its checkpoint")
    proof_checkpoint_resolved = Path(proof_checkpoint).expanduser().resolve()
    if proof_checkpoint_resolved != resume_resolved:
        raise ValueError(
            "--resume does not match the checkpoint recorded by --previous-stage-proof: "
            f"{resume_resolved} != {proof_checkpoint_resolved}"
        )
    return {
        "path": str(proof_resolved),
        "sha256": hashlib.sha256(proof_bytes).hexdigest(),
        "stage": expected_stage,
        "checkpoint": str(resume_resolved),
    }


def main() -> None:
    args = parse_args()
    try:
        previous_stage_proof = validate_previous_stage_proof(
            stage=args.stage,
            resume=args.resume,
            proof_path=args.previous_stage_proof,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    config = load_config()
    training = config["training"]
    vec_env = make_vec_env(
        lambda: T1GetUpEnv(stage=args.stage),
        n_envs=args.num_envs,
        seed=args.seed,
    )
    import torch

    policy_kwargs = {
        "net_arch": list(training["policy_layers"]),
        "activation_fn": torch.nn.ELU,
        "log_std_init": float(training["log_std_init"]),
    }
    if args.resume:
        model = PPO.load(str(args.resume), env=vec_env, device=args.device)
        reset_num_timesteps = False
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=float(training["learning_rate"]),
            n_steps=int(training["n_steps"]),
            batch_size=int(training["batch_size"]),
            n_epochs=int(training["n_epochs"]),
            gamma=float(training["gamma"]),
            gae_lambda=float(training["gae_lambda"]),
            ent_coef=float(training["ent_coef"]),
            clip_range=float(training["clip_range"]),
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            verbose=1,
            device=args.device,
        )
        reset_num_timesteps = True

    checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        checkpoint = checkpoint_dir / f"v039_stage_{args.stage:02d}_iter_{index:04d}.zip"
        if not checkpoint.exists():
            break
        index += 1
    start = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=reset_num_timesteps)
    elapsed = time.perf_counter() - start
    model.save(str(checkpoint.with_suffix("")))
    vec_env.close()
    payload = {
        "experiment": config["experiment"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "timesteps_requested": args.timesteps,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": args.device,
        "resume": str(args.resume) if args.resume else None,
        "previous_stage_proof": previous_stage_proof,
        "checkpoint": str(checkpoint),
        "elapsed_s": elapsed,
        "training_loop_completed": checkpoint.exists(),
        "quality_passed": False,
        "quality_note": "Training completion is not a quality pass; run fixed evaluation next.",
    }
    output = write_json(
        Path(__file__).resolve().parents[1]
        / "metrics"
        / f"{checkpoint.stem}_training.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
