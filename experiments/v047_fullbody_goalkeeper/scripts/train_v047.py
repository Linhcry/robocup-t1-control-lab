#!/usr/bin/env python3
"""Launch v047 PPO training without ever overwriting earlier run evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v047_fullbody_goalkeeper.rcss_goalkeeper_task import (  # noqa: E402
    TASK_ID,
    register_v047_task,
)
from experiments.v047_fullbody_goalkeeper.warmstart import (  # noqa: E402
    prepare_cached_warmstart,
    validate_engineering_source,
    validate_formal_source,
)


FORMAL_RESUME_SUPPORTED = False


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
        candidate = root / f"v047_training_launch_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v047 launch-manifest filename remains")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47000)
    parser.add_argument("--run-name", default="v047_fullbody_goalkeeper_full_r1")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warmstart-v045-checkpoint", type=Path)
    parser.add_argument("--v045-source-approval", type=Path)
    parser.add_argument("--engineering-source-proof", type=Path)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nan-guard", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        parser.error("num-envs and iterations must be positive")
    proof_args = (args.v045_source_approval, args.engineering_source_proof)
    if args.resume is not None and args.warmstart_v045_checkpoint is not None:
        parser.error("--resume and --warmstart-v045-checkpoint are mutually exclusive")
    if args.resume is not None and not args.engineering_smoke:
        parser.error(
            "formal v047 resume is fail-closed until an immutable v047 lineage "
            "proof is implemented; --resume is currently engineering-smoke only"
        )
    if args.warmstart_v045_checkpoint is None and any(
        item is not None for item in proof_args
    ):
        parser.error("warm-start proof arguments require --warmstart-v045-checkpoint")
    if args.warmstart_v045_checkpoint is not None:
        if args.engineering_smoke:
            if args.engineering_source_proof is None:
                parser.error(
                    "engineering warm-start requires --engineering-source-proof"
                )
            if args.v045_source_approval is not None:
                parser.error(
                    "engineering warm-start cannot consume a formal v045 approval"
                )
        else:
            if args.v045_source_approval is None:
                parser.error(
                    "formal warm-start requires --v045-source-approval"
                )
            if args.engineering_source_proof is not None:
                parser.error("formal warm-start cannot use an engineering source proof")
    if (
        args.resume is None
        and args.warmstart_v045_checkpoint is None
        and not args.engineering_smoke
    ):
        parser.error(
            "a formal v047 launch requires a v045 checkpoint bound to the terminal "
            "fixed-and-held-out source approval"
        )
    if args.engineering_smoke and not (
        args.cpu and args.iterations == 1 and 1 <= args.num_envs <= 4
    ):
        parser.error("engineering smoke is restricted to CPU, one iteration and 1-4 envs")
    return args


def main() -> int:
    args = _args()
    register_v047_task()
    from mjlab.scripts.train import TrainConfig, launch_training

    cfg = TrainConfig.from_task(TASK_ID)
    cfg.env.scene.num_envs = args.num_envs
    cfg.env.scene.terrain.num_envs = args.num_envs
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

    resume_proof = None
    warmstart_proof = None
    if args.resume is not None:
        checkpoint = args.resume.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        expected_root = (Path.cwd() / "logs" / "rsl_rl" / cfg.agent.experiment_name).resolve()
        try:
            checkpoint.relative_to(expected_root)
        except ValueError as error:
            raise ValueError(
                f"resume checkpoint must be inside {expected_root}, received {checkpoint}"
            ) from error
        cfg.agent.resume = True
        cfg.agent.load_run = f"^{checkpoint.parent.name}$"
        cfg.agent.load_checkpoint = f"^{checkpoint.name}$"
        resume_proof = {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        }
    elif args.warmstart_v045_checkpoint is not None:
        checkpoint = args.warmstart_v045_checkpoint.resolve()
        if args.engineering_smoke:
            source_binding = validate_engineering_source(
                checkpoint, args.engineering_source_proof
            )
        else:
            source_binding = validate_formal_source(
                checkpoint,
                args.v045_source_approval,
            )
        log_root = (Path.cwd() / "logs" / "rsl_rl" / cfg.agent.experiment_name).resolve()
        load_run, load_checkpoint, provenance_path = prepare_cached_warmstart(
            checkpoint,
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
    payload = {
        "schema_version": "v047_training_launch_v1",
        "task_id": TASK_ID,
        "num_envs": args.num_envs,
        "iterations_requested": args.iterations,
        "seed": args.seed,
        "run_name": args.run_name,
        "device_request": "cpu" if args.cpu else "cuda:0",
        "nan_guard": bool(args.nan_guard),
        "engineering_smoke": bool(args.engineering_smoke),
        "resume": resume_proof,
        "warmstart": warmstart_proof,
        "training_completed": False,
        "fixed_validation_completed": False,
        "held_out_test_completed": False,
        "quality_passed": False,
    }
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"launch_manifest": str(manifest)}, ensure_ascii=False))
    launch_training(TASK_ID, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
