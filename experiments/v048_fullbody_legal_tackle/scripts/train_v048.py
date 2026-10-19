"""Launch proof-bound v048 PPO without overwriting earlier evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v048_fullbody_legal_tackle.rcss_mjlab_task import (  # noqa: E402
    TASK_ID,
    register_v048_task,
)
from experiments.v048_fullbody_legal_tackle.warmstart import (  # noqa: E402
    prepare_cached_warmstart,
    validate_source_evidence,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_manifest() -> Path:
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = root / f"v048_training_launch_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v048 launch-manifest filename")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=48000)
    parser.add_argument("--run-name", default="v048_fullbody_legal_tackle_r1")
    parser.add_argument("--warmstart-v045-checkpoint", type=Path)
    parser.add_argument("--v045-source-approval", type=Path)
    parser.add_argument("--v045-training-summary", type=Path)
    parser.add_argument("--v045-candidate-proof", type=Path)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nan-guard", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        parser.error("num-envs and iterations must be positive")
    proof_args = (
        args.v045_source_approval,
        args.v045_training_summary,
        args.v045_candidate_proof,
    )
    if args.warmstart_v045_checkpoint is None:
        if any(item is not None for item in proof_args):
            parser.error("v045 proof arguments require --warmstart-v045-checkpoint")
        parser.error(
            "v048 training requires a proof-bound v045 warm-start checkpoint"
        )
    if args.engineering_smoke:
        if args.v045_source_approval is not None:
            parser.error(
                "engineering smoke uses only its non-formal summary/candidate chain"
            )
        if args.v045_training_summary is None or args.v045_candidate_proof is None:
            parser.error(
                "engineering warm-start requires --v045-training-summary and "
                "--v045-candidate-proof"
            )
        if not args.cpu or args.iterations != 1:
            parser.error(
                "engineering warm-start is restricted to CPU and exactly one PPO update"
            )
    else:
        if args.v045_source_approval is None:
            parser.error(
                "formal v048 training requires --v045-source-approval from the "
                "unified v045 downstream approval gate"
            )
        if args.v045_training_summary is not None or args.v045_candidate_proof is not None:
            parser.error(
                "formal v048 training accepts only --v045-source-approval, not "
                "direct summary/candidate evidence"
            )
        if (
            args.num_envs != 4096
            or args.iterations != 3000
            or args.seed != 48000
            or args.cpu
            or not args.nan_guard
        ):
            parser.error(
                "formal v048 training requires 4096 envs, 3000 iterations, seed "
                "48000, CUDA, and --nan-guard"
            )
    return args


def main() -> int:
    args = _args()
    register_v048_task()
    from mjlab.scripts.train import TrainConfig, launch_training

    cfg = TrainConfig.from_task(TASK_ID)
    cfg.env.scene.num_envs = args.num_envs
    cfg.env.seed = args.seed
    cfg.agent.seed = args.seed
    cfg.agent.max_iterations = args.iterations
    cfg.agent.run_name = args.run_name
    cfg.agent.logger = "tensorboard"
    cfg = replace(
        cfg,
        gpu_ids=None if args.cpu else [0],
        enable_nan_guard=args.nan_guard,
    )

    warmstart_proof = None
    if args.warmstart_v045_checkpoint is not None:
        source_binding = validate_source_evidence(
            args.warmstart_v045_checkpoint,
            args.v045_training_summary,
            args.v045_candidate_proof,
            approval_manifest_path=args.v045_source_approval,
            engineering_smoke=args.engineering_smoke,
        )
        log_root = (
            Path.cwd() / "logs" / "rsl_rl" / cfg.agent.experiment_name
        ).resolve()
        load_run, load_checkpoint, provenance_path = prepare_cached_warmstart(
            args.warmstart_v045_checkpoint,
            source_binding,
            log_root,
            learning_rate=cfg.agent.algorithm.learning_rate,
        )
        cfg.agent.resume = True
        cfg.agent.load_run = f"^{load_run}$"
        cfg.agent.load_checkpoint = f"^{load_checkpoint}$"
        warmstart_proof = {
            "path": str(provenance_path),
            "sha256": _sha256(provenance_path),
            "source_binding_mode": source_binding["mode"],
            "formal_source_binding": source_binding["mode"] == "formal",
        }

    manifest = _unique_manifest()
    manifest_payload = {
        "schema_version": "v048_training_launch_v1",
        "task_id": TASK_ID,
        "num_environments": args.num_envs,
        "iterations_requested": args.iterations,
        "seed": args.seed,
        "run_name": args.run_name,
        "device_request": "cpu" if args.cpu else "cuda:0",
        "nan_guard": bool(args.nan_guard),
        "engineering_smoke": bool(args.engineering_smoke),
        "warmstart": warmstart_proof,
        "training_completed": False,
        "fixed_validation_completed": False,
        "held_out_test_completed": False,
        "quality_passed": False,
    }
    manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"launch_manifest": str(manifest)}, ensure_ascii=False))
    launch_training(TASK_ID, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
