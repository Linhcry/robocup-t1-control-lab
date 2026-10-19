#!/usr/bin/env python3
"""Create an immutable, explicitly non-promotable v045 source proof for smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v047_fullbody_goalkeeper.warmstart import sha256  # noqa: E402


def _unique_output() -> Path:
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        path = root / f"v047_v045_engineering_source_proof_run_{run:02d}.json"
        if not path.exists():
            return path
    raise RuntimeError("no unused engineering source-proof filename remains")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--reason",
        default=(
            "Formal v045 training summary and target-candidate proof are not yet "
            "available; this checkpoint is used only to test transfer plumbing."
        ),
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = _unique_output()
    payload = {
        "schema_version": "v047_v045_engineering_source_v1",
        "source_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
        },
        "reason": args.reason,
        "engineering_smoke_only": True,
        "formal_v045_training_summary_bound": False,
        "formal_v045_candidate_proof_bound": False,
        "promotable": False,
        "quality_passed": False,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "quality_passed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

