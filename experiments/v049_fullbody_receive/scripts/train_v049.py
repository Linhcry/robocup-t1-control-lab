#!/usr/bin/env python3
"""Train v049; optional CPU one-update evidence is always non-promotable."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v049_fullbody_receive.rcss_mjlab_task import (  # noqa: E402
    TASK_ID,
    register_v049_task,
)
from experiments.v045_rcss_t1_getup_adaptation.source_approval import (  # noqa: E402
    validate_source_approval,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _next_output(*, warm_start: bool) -> Path:
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        stem = "v049_warmstart_cpu_ppo_smoke" if warm_start else "v049_cpu_ppo_smoke"
        candidate = root / f"{stem}_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v049 PPO smoke evidence filename")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=49000)
    parser.add_argument("--run-name", default="v049_fullbody_receive_full_r1")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nan-guard", action="store_true")
    parser.add_argument("--write-smoke-evidence", action="store_true")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--warm-start-proof", type=Path)
    parser.add_argument("--v045-source-approval", type=Path)
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        parser.error("num-envs and iterations must be positive")
    if args.write_smoke_evidence and not (
        args.cpu and args.iterations == 1 and 1 <= args.num_envs <= 4
    ):
        parser.error("smoke evidence requires CPU, one iteration and 1-4 envs")
    if (args.warm_start is None) != (args.warm_start_proof is None):
        parser.error("--warm-start and --warm-start-proof must be supplied together")
    if args.engineering_smoke and not (
        args.warm_start is not None
        and args.write_smoke_evidence
        and args.cpu
        and args.iterations == 1
        and 1 <= args.num_envs <= 4
    ):
        parser.error(
            "--engineering-smoke requires a warm-start and the restricted CPU one-update smoke protocol"
        )
    if args.engineering_smoke:
        if args.v045_source_approval is not None:
            parser.error("engineering smoke cannot consume a formal source approval")
    else:
        if args.warm_start is None or args.v045_source_approval is None:
            parser.error(
                "formal v049 training requires a warm-start checkpoint, its provenance and --v045-source-approval"
            )
    return args


def _validate_warm_start(
    checkpoint: Path,
    proof_path: Path,
    engineering: bool,
    source_approval_path: Path | None = None,
) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    proof_path = proof_path.expanduser().resolve()
    if not checkpoint.is_file() or not proof_path.is_file():
        raise FileNotFoundError(f"warm-start artifact/proof missing: {checkpoint}, {proof_path}")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != "v049_v045_shape_aware_warmstart_v1":
        raise ValueError("v049 warm-start proof schema mismatch")
    checkpoint_sha = _sha256(checkpoint)
    recorded = proof.get("output_checkpoint", {})
    if (
        Path(recorded.get("path", "")).expanduser().resolve() != checkpoint
        or recorded.get("sha256") != checkpoint_sha
    ):
        raise ValueError("warm-start checkpoint does not match its provenance proof")
    checks = proof.get("mapping_checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise ValueError("warm-start mapping checks are incomplete")
    formal = proof.get("formal_source_approved") is True
    if engineering:
        if source_approval_path is not None:
            raise ValueError("engineering warm-start cannot consume a formal approval")
        if formal or proof.get("engineering_smoke_only") is not True:
            raise ValueError("engineering smoke requires explicitly non-formal provenance")
        source_binding = None
    else:
        if source_approval_path is None:
            raise ValueError("formal warm-start requires a v045 source approval")
        if not formal or proof.get("formal_training_allowed") is not True:
            raise ValueError(
                "formal training requires fixed-and-held-out-approved v045 provenance"
            )
        source_ref = proof.get("source_checkpoint")
        if not isinstance(source_ref, dict):
            raise ValueError("warm-start provenance has no v045 source checkpoint")
        source_checkpoint = Path(str(source_ref.get("path", ""))).expanduser().resolve()
        source_binding = validate_source_approval(
            source_checkpoint, source_approval_path
        )
        approval_ref = proof.get("v045_source_approval")
        approval_path = source_approval_path.expanduser().resolve()
        if (
            not isinstance(approval_ref, dict)
            or Path(str(approval_ref.get("path", ""))).expanduser().resolve()
            != approval_path
            or approval_ref.get("sha256") != _sha256(approval_path)
        ):
            raise ValueError("warm-start provenance binds a different source approval")
        if source_binding["source_checkpoint"]["sha256"] != source_ref.get("sha256"):
            raise ValueError("approved checkpoint differs from warm-start source")
    recorded_sources = proof.get("source_code_sha256")
    if not isinstance(recorded_sources, dict) or not recorded_sources:
        raise ValueError("warm-start proof has no source-code hash manifest")
    for relative, expected_sha in recorded_sources.items():
        source = REPO_ROOT / relative
        if not source.is_file() or _sha256(source) != expected_sha:
            raise ValueError(f"warm-start source drift detected: {relative}")
    return {
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "proof": {"path": str(proof_path), "sha256": _sha256(proof_path)},
        "formal_source_approved": formal,
        "engineering_smoke_only": not formal,
        "source_binding": source_binding,
        "v045_source_approval": proof.get("v045_source_approval"),
        "source_checkpoint": proof.get("source_checkpoint"),
        "v045_training_summary": proof.get("v045_training_summary"),
        "v045_candidate_proof": proof.get("v045_candidate_proof"),
        "mapping_checks": checks,
    }


def _stage_warm_start(checkpoint: Path, proof_path: Path, log_root: Path) -> tuple[str, str]:
    checkpoint = checkpoint.resolve()
    proof_path = proof_path.resolve()
    run_name = f"warmstart_{_sha256(checkpoint)[:16]}_{_sha256(proof_path)[:12]}"
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    staged = run_dir / checkpoint.name
    staged_proof = run_dir / proof_path.name
    for source, target in ((checkpoint, staged), (proof_path, staged_proof)):
        if target.exists():
            if _sha256(target) != _sha256(source):
                raise RuntimeError(f"existing staged warm-start differs: {target}")
        else:
            shutil.copy2(source, target)
    return run_name, staged.name


def main() -> int:
    args = _parse_args()
    warm_start = None
    if args.warm_start is not None:
        warm_start = _validate_warm_start(
            args.warm_start,
            args.warm_start_proof,
            args.engineering_smoke,
            args.v045_source_approval,
        )
    register_v049_task()
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

    log_root = Path.cwd() / "logs" / "rsl_rl" / cfg.agent.experiment_name
    if warm_start is not None:
        load_run, load_checkpoint = _stage_warm_start(
            args.warm_start, args.warm_start_proof, log_root
        )
        cfg.agent.resume = True
        cfg.agent.load_run = f"^{load_run}$"
        cfg.agent.load_checkpoint = f"^{load_checkpoint}$"
    before = set(log_root.glob("*")) if log_root.exists() else set()
    launch_training(TASK_ID, cfg)

    if args.write_smoke_evidence:
        created = sorted(path for path in set(log_root.glob("*")) - before if path.is_dir())
        if len(created) != 1:
            raise RuntimeError(f"expected exactly one new PPO log directory, found {created}")
        checkpoints = sorted(created[0].glob("model_*.pt"))
        if not checkpoints:
            raise RuntimeError("v049 CPU PPO smoke produced no checkpoint")
        checkpoint = checkpoints[-1]
        output = _next_output(warm_start=warm_start is not None)
        import torch

        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        output_optimizer_entries = len(
            checkpoint_payload["optimizer_state_dict"]["state"]
        )
        output_counter = int(
            checkpoint_payload.get("infos", {})
            .get("env_state", {})
            .get("common_step_counter", -1)
        )
        payload = {
            "schema_version": (
                "v049_v045_warmstart_cpu_ppo_smoke_v1"
                if warm_start is not None
                else "v049_cpu_ppo_smoke_v1"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": TASK_ID,
            "device": "cpu",
            "num_envs": args.num_envs,
            "seed": args.seed,
            "requested_iterations": args.iterations,
            "ppo_update_completed": True,
            "strict_checkpoint_load_completed": warm_start is not None,
            "warm_start": warm_start,
            "log_directory": str(created[0].resolve()),
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": _sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "iteration": int(checkpoint_payload["iter"]),
                "common_step_counter": output_counter,
                "optimizer_state_entries": output_optimizer_entries,
            },
            "fresh_training_state_checks": {
                "loaded_iteration_started_at_zero": bool(
                    warm_start is not None and int(checkpoint_payload["iter"]) == 0
                )
                if warm_start is not None
                else None,
                "one_rollout_advanced_counter_to_32": bool(
                    warm_start is not None and output_counter == 32
                )
                if warm_start is not None
                else None,
                "optimizer_populated_only_after_new_update": bool(
                    warm_start is not None and output_optimizer_entries > 0
                )
                if warm_start is not None
                else None,
            },
            "training_completed": False,
            "fixed_validation_passed": False,
            "heldout_validation_passed": False,
            "quality_passed": False,
            "conclusion": (
                "One CPU PPO update proves only that the v049 training pipeline runs; "
                "it is not receiving ability evidence."
            ),
        }
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"smoke_evidence": str(output), "quality_passed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
