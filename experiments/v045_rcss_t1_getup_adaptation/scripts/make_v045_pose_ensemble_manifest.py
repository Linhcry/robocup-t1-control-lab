"""Create an immutable, hash-bound v045 pose-routed ensemble manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


POSE_NAMES = ("prone", "supine", "left_side", "right_side")


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
    for pose in POSE_NAMES:
        parser.add_argument(f"--{pose.replace('_', '-')}", type=Path, required=True)
        parser.add_argument(
            f"--{pose.replace('_', '-')}-action-scale",
            type=float,
            default=0.6,
        )
    parser.add_argument("--fallback-pose", choices=POSE_NAMES, default="right_side")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for pose in POSE_NAMES:
        scale = getattr(args, f"{pose}_action_scale")
        if not math.isfinite(scale) or not 0.0 < scale <= 0.6:
            parser.error(f"{pose} action scale must be finite and in (0, 0.6]")
    return args


def main() -> int:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite ensemble manifest")
    policies = {
        pose: {
            **_reference(getattr(args, pose)),
            "action_scale_rad": float(getattr(args, f"{pose}_action_scale")),
        }
        for pose in POSE_NAMES
    }
    payload = {
        "schema_version": "v045_pose_routed_ensemble_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "routing": (
            "select once from initial torso-frame projected gravity: dominant +x "
            "prone, -x supine, +y left_side, -y right_side; use fallback below "
            "0.08 horizontal norm; each expert retains raw-action recurrence and "
            "uses its manifest-bound deployment action scale"
        ),
        "fallback_pose": args.fallback_pose,
        "policies": policies,
        "quality_passed": False,
        "scope": "Engineering candidate only; exact RCSS evaluation is required.",
        "script": _reference(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), **payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
