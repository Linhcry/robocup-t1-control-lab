"""Create an auditable v045 checkpoint with a scaled actor output layer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch


ACTOR_OUTPUT_WEIGHT = "mlp.6.weight"
ACTOR_OUTPUT_BIAS = "mlp.6.bias"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.factor) or not 0.0 < args.factor <= 1.0:
        parser.error("factor must be finite and in (0, 1]")
    return args


def main() -> int:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    evidence = args.evidence.expanduser().resolve()
    if output.exists() or evidence.exists():
        raise FileExistsError("refusing to overwrite checkpoint or evidence")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    actor = payload.get("actor_state_dict")
    if not isinstance(actor, dict):
        raise ValueError("checkpoint is missing actor_state_dict")
    for key in (ACTOR_OUTPUT_WEIGHT, ACTOR_OUTPUT_BIAS):
        value = actor.get(key)
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"checkpoint has an invalid actor output tensor: {key}")
    if actor[ACTOR_OUTPUT_WEIGHT].shape[0] != 23:
        raise ValueError("actor output layer is not the canonical 23D T1 action")

    transformed = copy.deepcopy(payload)
    transformed_actor = transformed["actor_state_dict"]
    for key in (ACTOR_OUTPUT_WEIGHT, ACTOR_OUTPUT_BIAS):
        transformed_actor[key] = transformed_actor[key] * float(args.factor)
    optimizer = transformed.get("optimizer_state_dict")
    if isinstance(optimizer, dict):
        optimizer["state"] = {}
    transformed["iter"] = 0
    transformed["infos"] = {
        "v045_actor_output_calibration": {
            "source": _reference(source),
            "factor": float(args.factor),
            "scaled_tensors": [ACTOR_OUTPUT_WEIGHT, ACTOR_OUTPUT_BIAS],
            "equivalent_deployment_action_scale_rad": 0.6 * float(args.factor),
        }
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(transformed, output)
    proof = {
        "schema_version": "v045_scaled_actor_checkpoint_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": _reference(source),
        "factor": float(args.factor),
        "scaled_tensors": [ACTOR_OUTPUT_WEIGHT, ACTOR_OUTPUT_BIAS],
        "checkpoint": _reference(output),
        "quality_passed": False,
        "scope": "Engineering candidate only; exact RCSS evaluation is required.",
        "script": _reference(Path(__file__).resolve()),
    }
    evidence.parent.mkdir(parents=True, exist_ok=True)
    with evidence.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"evidence": str(evidence), **proof}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
