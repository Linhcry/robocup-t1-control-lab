#!/usr/bin/env python3
"""Create one immutable, shape-aware v045 -> v046 warm-start checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v046_fullbody_pass.warmstart import transform_checkpoint  # noqa: E402
from experiments.v045_rcss_t1_getup_adaptation.source_approval import (  # noqa: E402
    validate_source_approval,
)


def _next_target() -> Path:
    root = EXPERIMENT_ROOT / "artifacts"
    for run in range(1, 10_000):
        path = root / f"v046_v045_shape_warmstart_run_{run:02d}.pt"
        if not path.exists():
            return path
    raise RuntimeError("no unused v046 warm-start artifact filename")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-approval", type=Path)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    args = parser.parse_args()
    if args.engineering_smoke:
        if args.source_approval is not None:
            parser.error("engineering smoke cannot consume a formal source approval")
        source_binding = {
            "mode": "engineering_smoke",
            "formal_source_approved": False,
            "promotable_source_binding": False,
        }
    else:
        if args.source_approval is None:
            parser.error("formal warm-start creation requires --source-approval")
        source_binding = validate_source_approval(args.source, args.source_approval)
    output = args.output.resolve() if args.output else _next_target().resolve()
    result = transform_checkpoint(
        args.source, output, learning_rate=args.learning_rate
    )
    result.update(
        {
            "schema_version": "v046_warmstart_creation_v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_binding": source_binding,
            "formal_source_approved": not args.engineering_smoke,
            "engineering_smoke_only": bool(args.engineering_smoke),
            "formal_training_allowed": not args.engineering_smoke,
            "training_completed": False,
            "validation_passed": False,
            "quality_passed": False,
        }
    )
    proof = output.with_suffix(".provenance.json")
    if proof.exists():
        raise FileExistsError(proof)
    proof.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checkpoint": str(output), "proof": str(proof)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
