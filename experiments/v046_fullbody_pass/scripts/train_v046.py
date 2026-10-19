#!/usr/bin/env python3
"""Train v046; CPU smoke evidence is opt-in and always non-promotable."""

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

from experiments.v046_fullbody_pass.rcss_mjlab_task import (  # noqa: E402
    TASK_ID,
    register_v046_task,
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


def _next_output() -> Path:
    metrics = EXPERIMENT_ROOT / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = metrics / f"v046_cpu_ppo_smoke_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v046 CPU PPO smoke evidence filename")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=46000)
    parser.add_argument("--run-name", default="v046_directional_pass_full_r1")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nan-guard", action="store_true")
    parser.add_argument("--write-smoke-evidence", action="store_true")
    parser.add_argument(
        "--warmstart-checkpoint",
        type=Path,
        help="already transformed 84/101-dimensional v046 checkpoint",
    )
    parser.add_argument(
        "--warmstart-audit",
        type=Path,
        help="passing immutable v045->v046 shape-mapping audit",
    )
    parser.add_argument("--v045-source-approval", type=Path)
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        parser.error("num-envs and iterations must be positive")
    if args.write_smoke_evidence and not (
        args.cpu and args.iterations == 1 and 1 <= args.num_envs <= 4
    ):
        parser.error(
            "smoke evidence is restricted to CPU, one iteration and 1-4 envs"
        )
    if (args.warmstart_checkpoint is None) != (args.warmstart_audit is None):
        parser.error("--warmstart-checkpoint and --warmstart-audit are required together")
    if args.engineering_smoke:
        if not (
            args.cpu
            and args.iterations == 1
            and 1 <= args.num_envs <= 4
            and args.write_smoke_evidence
        ):
            parser.error(
                "engineering smoke is restricted to CPU, one iteration, 1-4 envs and smoke evidence"
            )
        if args.v045_source_approval is not None:
            parser.error("engineering smoke cannot consume a formal source approval")
    else:
        if args.warmstart_checkpoint is None or args.v045_source_approval is None:
            parser.error(
                "formal v046 training requires a warm-start checkpoint, its audit and --v045-source-approval"
            )
    return args


def _validate_warmstart_audit(
    checkpoint: Path,
    audit_path: Path,
    source_approval: Path | None,
    *,
    engineering_smoke: bool,
) -> dict:
    checkpoint = checkpoint.resolve()
    audit_path = audit_path.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    proof = json.loads(audit_path.read_text(encoding="utf-8"))
    checks = proof.get("checks")
    schema = proof.get("schema_version")
    allowed_schemas = (
        {"v046_v045_shape_warmstart_audit_v1", "v046_v045_shape_warmstart_audit_v2"}
        if engineering_smoke
        else {"v046_v045_shape_warmstart_audit_v2"}
    )
    if schema not in allowed_schemas:
        raise ValueError("warm-start audit schema mismatch")
    if proof.get("warmstart_semantics_passed") is not True:
        raise ValueError("warm-start audit did not pass")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("warm-start audit checks are incomplete or failed")
    checkpoint_sha = _sha256(checkpoint)
    if proof.get("target_checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("warm-start checkpoint hash differs from its audit")
    recorded_sources = proof.get("sources", {})
    if not isinstance(recorded_sources, dict) or not recorded_sources:
        raise ValueError("warm-start audit has no source-code hash manifest")
    for relative, expected_sha in recorded_sources.items():
        current = EXPERIMENT_ROOT / relative
        if not current.is_file() or _sha256(current) != expected_sha:
            raise ValueError(f"warm-start implementation changed after audit: {relative}")

    source_binding = proof.get("source_binding")
    if engineering_smoke:
        if schema == "v046_v045_shape_warmstart_audit_v2" and (
            proof.get("engineering_smoke_only") is not True
            or proof.get("formal_source_approved") is not False
        ):
            raise ValueError("engineering warm-start audit is not explicitly non-formal")
    else:
        if source_approval is None:
            raise ValueError("formal warm-start validation requires a source approval")
        if proof.get("formal_source_approved") is not True or proof.get(
            "formal_training_allowed"
        ) is not True:
            raise ValueError("warm-start audit is not approved for formal training")
        source_ref = proof.get("source_checkpoint")
        if not isinstance(source_ref, dict):
            raise ValueError("warm-start audit has no v045 source checkpoint")
        source_checkpoint = Path(str(source_ref.get("path", ""))).expanduser().resolve()
        binding = validate_source_approval(source_checkpoint, source_approval)
        approval_ref = proof.get("v045_source_approval")
        if (
            not isinstance(approval_ref, dict)
            or Path(str(approval_ref.get("path", ""))).expanduser().resolve()
            != source_approval.expanduser().resolve()
            or approval_ref.get("sha256") != _sha256(source_approval.expanduser().resolve())
        ):
            raise ValueError("warm-start audit binds a different v045 source approval")
        if binding.get("source_checkpoint", {}).get("sha256") != source_ref.get("sha256"):
            raise ValueError("approved v045 checkpoint differs from the warm-start source")
        source_binding = binding
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "audit": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "source_checkpoint": proof.get("source_checkpoint"),
        "source_binding": source_binding,
        "formal_source_approved": not engineering_smoke,
        "engineering_smoke_only": engineering_smoke,
        "mapping_contract": proof.get("mapping_contract"),
    }


def _stage_warmstart(checkpoint: Path, log_root: Path, proof: dict) -> tuple[str, str]:
    """Place an audited immutable checkpoint where the RSL loader expects it."""

    # The same transformed tensor file may be re-audited after surrounding
    # source changes.  Include both hashes so an older proof can never be
    # silently replaced inside an existing staging directory.
    run_name = (
        f"bootstrap_{proof['checkpoint_sha256'][:16]}_"
        f"{proof['audit_sha256'][:12]}_v045_shape_v1"
    )
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "v046_v045_shape_warmstart.pt"
    if target.exists():
        if _sha256(target) != proof["checkpoint_sha256"]:
            raise RuntimeError(f"existing staged warm start has a hash mismatch: {target}")
    else:
        shutil.copy2(checkpoint, target)
    stage_proof = run_dir / "bootstrap_provenance.json"
    expected = {
        "schema_version": "v046_staged_warmstart_v1",
        **proof,
        "staged_checkpoint": str(target.resolve()),
        "staged_sha256": _sha256(target),
    }
    if stage_proof.exists():
        if json.loads(stage_proof.read_text(encoding="utf-8")) != expected:
            raise RuntimeError(f"existing staged proof differs: {stage_proof}")
    else:
        stage_proof.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return run_name, target.name


def main() -> int:
    args = _parse_args()
    register_v046_task()
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

    warmstart_proof = None
    if args.warmstart_checkpoint is not None:
        warmstart_proof = _validate_warmstart_audit(
            args.warmstart_checkpoint,
            args.warmstart_audit,
            args.v045_source_approval,
            engineering_smoke=args.engineering_smoke,
        )
        log_root = Path.cwd() / "logs" / "rsl_rl" / cfg.agent.experiment_name
        load_run, load_checkpoint = _stage_warmstart(
            args.warmstart_checkpoint.resolve(), log_root, warmstart_proof
        )
        cfg.agent.resume = True
        cfg.agent.load_run = f"^{load_run}$"
        cfg.agent.load_checkpoint = f"^{load_checkpoint}$"

    log_root = Path.cwd() / "logs" / "rsl_rl" / cfg.agent.experiment_name
    before = set(log_root.glob("*")) if log_root.exists() else set()
    launch_training(TASK_ID, cfg)

    if args.write_smoke_evidence:
        after = set(log_root.glob("*"))
        created = sorted(path for path in after - before if path.is_dir())
        if len(created) != 1:
            raise RuntimeError(f"expected one new log directory, found {created}")
        run_dir = created[0]
        checkpoints = sorted(run_dir.glob("model_*.pt"))
        if not checkpoints:
            raise RuntimeError(f"PPO smoke produced no checkpoint in {run_dir}")
        checkpoint = checkpoints[-1]
        import torch

        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        actor_state = checkpoint_payload.get("actor_state_dict", {})
        critic_state = checkpoint_payload.get("critic_state_dict", {})
        checkpoint_contract_checks = {
            "actor_first_layer_is_512x84": tuple(
                actor_state.get("mlp.0.weight", torch.empty(0)).shape
            )
            == (512, 84),
            "critic_first_layer_is_512x101": tuple(
                critic_state.get("mlp.0.weight", torch.empty(0)).shape
            )
            == (512, 101),
            "action_distribution_is_23d": tuple(
                actor_state.get("distribution.log_std_param", torch.empty(0)).shape
            )
            == (23,),
            "checkpoint_tensors_finite": all(
                torch.isfinite(value).all().item()
                for state in (actor_state, critic_state)
                for value in state.values()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            ),
        }
        if not all(checkpoint_contract_checks.values()):
            raise RuntimeError(
                f"post-update checkpoint contract failed: {checkpoint_contract_checks}"
            )
        output = _next_output()
        payload = {
            "schema_version": "v046_cpu_ppo_smoke_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": TASK_ID,
            "device": "cpu",
            "num_envs": args.num_envs,
            "seed": args.seed,
            "requested_iterations": args.iterations,
            "ppo_update_completed": True,
            "log_directory": str(run_dir.resolve()),
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": _sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            },
            "warmstart": warmstart_proof,
            "strict_checkpoint_load_completed": bool(warmstart_proof is not None),
            "post_update_checkpoint_checks": checkpoint_contract_checks,
            "training_completed": False,
            "validation_passed": False,
            "quality_passed": False,
            "conclusion": (
                "One CPU PPO update proves the v046 training path is executable. "
                "It is not evidence of passing or receiving skill."
            ),
        }
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"smoke_evidence": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
