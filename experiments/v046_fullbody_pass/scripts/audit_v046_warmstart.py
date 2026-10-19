#!/usr/bin/env python3
"""Independently audit one shape-aware v045 -> v046 warm start."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v046_fullbody_pass.warmstart import (  # noqa: E402
    TRANSFORM_VERSION,
    audit_mapping,
    sha256_file,
)
from experiments.v045_rcss_t1_getup_adaptation.source_approval import (  # noqa: E402
    validate_source_approval,
)


def _next_output() -> Path:
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        path = root / f"v046_v045_warmstart_audit_run_{run:02d}.json"
        if not path.exists():
            return path
    raise RuntimeError("no unused v046 warm-start audit filename")


def _all_true(value: object) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        item is True for item in value.values()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-approval", type=Path)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument(
        "--cpu-smoke-proof",
        type=Path,
        help="optional completed strict-load and one-update CPU evidence",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.engineering_smoke and args.source_approval is not None:
        parser.error("engineering audit cannot consume a formal source approval")
    if not args.engineering_smoke and args.source_approval is None:
        parser.error("formal warm-start audit requires --source-approval")
    source = args.source.resolve()
    target = args.target.resolve()
    for path in (source, target):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve() if args.output else _next_output()
    if output.exists():
        raise FileExistsError(output)
    source_binding = (
        {
            "mode": "engineering_smoke",
            "formal_source_approved": False,
            "promotable_source_binding": False,
        }
        if args.engineering_smoke
        else validate_source_approval(source, args.source_approval)
    )
    checks = audit_mapping(source, target)
    checks["formal_source_approval_revalidated"] = bool(
        args.engineering_smoke
        or source_binding.get("formal_source_approved") is True
    )
    smoke_binding = None
    if args.cpu_smoke_proof is not None:
        smoke_path = args.cpu_smoke_proof.resolve()
        if not smoke_path.is_file():
            raise FileNotFoundError(smoke_path)
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        smoke_binding = {"path": str(smoke_path), "sha256": sha256_file(smoke_path)}
        checks.update(
            {
                "cpu_smoke_schema_valid": smoke.get("schema_version")
                == "v046_cpu_ppo_smoke_v1",
                "cpu_strict_checkpoint_load_completed": smoke.get(
                    "strict_checkpoint_load_completed"
                )
                is True,
                "cpu_one_ppo_update_completed": smoke.get("ppo_update_completed")
                is True,
                "cpu_smoke_used_audited_warmstart": smoke.get("warmstart", {}).get(
                    "checkpoint_sha256"
                )
                == sha256_file(target),
                "cpu_post_update_shapes_and_finiteness_passed": _all_true(
                    smoke.get("post_update_checkpoint_checks")
                ),
                "cpu_smoke_quality_remained_false": smoke.get("quality_passed")
                is False,
            }
        )
    passed = bool(checks) and all(checks.values())
    source_files = {}
    for path in sorted(EXPERIMENT_ROOT.rglob("*.py")):
        if "__pycache__" not in path.parts:
            source_files[str(path.relative_to(EXPERIMENT_ROOT))] = sha256_file(path)
    payload = {
        "schema_version": "v046_v045_shape_warmstart_audit_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "transform": TRANSFORM_VERSION,
        "source_checkpoint": {"path": str(source), "sha256": sha256_file(source)},
        "target_checkpoint": {"path": str(target), "sha256": sha256_file(target)},
        "source_binding": source_binding,
        "v045_source_approval": source_binding.get("v045_source_approval"),
        "cpu_smoke_proof": smoke_binding,
        "mapping_contract": {
            "actor_shared_columns": [0, 75],
            "actor_new_zero_columns": [75, 84],
            "critic_shared_columns": [0, 75],
            "critic_base_lin_source_columns": [75, 78],
            "critic_base_lin_target_columns": [84, 87],
            "critic_new_zero_columns": [[75, 84], [87, 101]],
            "subsequent_layers_and_action_distribution": "exact_copy",
            "optimizer_iteration_curriculum": "reset",
        },
        "sources": source_files,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(source_files, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "checks": checks,
        "warmstart_semantics_passed": passed,
        "formal_source_approved": bool(
            not args.engineering_smoke
            and source_binding.get("formal_source_approved") is True
        ),
        "engineering_smoke_only": bool(args.engineering_smoke),
        "formal_training_allowed": bool(
            not args.engineering_smoke
            and source_binding.get("formal_source_approved") is True
            and passed
        ),
        "checkpoint_load_test_passed": bool(
            smoke_binding is not None
            and checks.get("cpu_strict_checkpoint_load_completed", False)
        ),
        "ppo_update_completed": bool(
            smoke_binding is not None
            and checks.get("cpu_one_ppo_update_completed", False)
        ),
        "training_completed": False,
        "validation_passed": False,
        "quality_passed": False,
        "conclusion": (
            "The network mapping is exact and safe to test on CPU; no pass skill is claimed."
            if passed
            else "Warm-start mapping audit failed; loading or training is blocked."
        ),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "warmstart_semantics_passed": passed,
                "quality_passed": False,
                "failed_checks": [name for name, ok in checks.items() if not ok],
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
