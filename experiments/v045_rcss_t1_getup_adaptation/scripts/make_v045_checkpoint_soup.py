"""Create auditable linear weight-soup candidates between two v045 policies."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch


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
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.alphas)) != len(args.alphas):
        parser.error("alphas must be unique")
    if any(not math.isfinite(alpha) or not 0.0 < alpha < 1.0 for alpha in args.alphas):
        parser.error("every alpha must be finite and strictly between 0 and 1")
    return args


def _interpolate_state(
    source: dict[str, torch.Tensor],
    adapted: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if set(source) != set(adapted):
        raise ValueError("checkpoint network keys differ")
    result: dict[str, torch.Tensor] = {}
    for key in source:
        left = source[key]
        right = adapted[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"checkpoint tensor contract differs: {key}")
        if left.is_floating_point():
            value = left.to(torch.float64).lerp(right.to(torch.float64), alpha)
            result[key] = value.to(dtype=left.dtype)
        else:
            # Normalizer sample counters are integer tensors and cannot be
            # meaningfully interpolated.  Keep the adapted counter while all
            # learned floating statistics and network weights are blended.
            result[key] = right.clone()
    return result


def main() -> int:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    adapted = args.adapted.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    evidence = args.evidence.expanduser().resolve()
    if output_dir.exists() or evidence.exists():
        raise FileExistsError("refusing to overwrite soup artifacts or evidence")
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    adapted_payload = torch.load(adapted, map_location="cpu", weights_only=False)
    for name in ("actor_state_dict", "critic_state_dict"):
        if name not in source_payload or name not in adapted_payload:
            raise ValueError(f"checkpoint is missing {name}")

    output_dir.mkdir(parents=True, exist_ok=False)
    candidates: list[dict[str, object]] = []
    for alpha in args.alphas:
        payload = copy.deepcopy(source_payload)
        for name in ("actor_state_dict", "critic_state_dict"):
            payload[name] = _interpolate_state(
                source_payload[name], adapted_payload[name], float(alpha)
            )
        payload["optimizer_state_dict"]["state"] = {}
        payload["iter"] = 0
        payload["infos"] = {
            "v045_weight_soup": {
                "source": _reference(source),
                "adapted": _reference(adapted),
                "alpha_adapted": float(alpha),
            }
        }
        tag = f"{int(round(alpha * 100)):03d}"
        output = output_dir / f"model_soup_alpha_{tag}.pt"
        torch.save(payload, output)
        candidates.append(
            {
                "alpha_adapted": float(alpha),
                "checkpoint": _reference(output),
            }
        )

    script = Path(__file__).resolve()
    proof = {
        "schema_version": "v045_checkpoint_soup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": _reference(source),
        "adapted": _reference(adapted),
        "interpolation": (
            "linear_float64_then_cast_for_floating_actor_and_critic_tensors; "
            "adapted_value_for_nonfloating_normalizer_counters"
        ),
        "candidates": candidates,
        "quality_passed": False,
        "scope": "Engineering candidates only; exact RCSS evaluation is required.",
        "script": _reference(script),
    }
    evidence.parent.mkdir(parents=True, exist_ok=True)
    with evidence.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"evidence": str(evidence), **proof}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
