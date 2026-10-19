"""Require the latest genuinely promoted locomotion/ball-control prerequisite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMOTED_V038_METRICS = (
    PROJECT_ROOT
    / "experiments"
    / "v038_high_speed_dribble"
    / "metrics"
    / "v038_stage_00_iter_0002_held_out_test.json"
)


def team_training_gate() -> tuple[bool, str, dict[str, Any] | None]:
    """Return whether the frozen v038 held-out gate allows team experiments."""

    if not PROMOTED_V038_METRICS.exists():
        return False, f"missing prerequisite: {PROMOTED_V038_METRICS}", None
    payload = json.loads(PROMOTED_V038_METRICS.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    passed = bool(
        payload.get("quality_passed")
        and summary.get("all_23_targets_sent")
        and float(summary.get("fall_rate", 1.0)) == 0.0
        and float(summary.get("lost_possession_rate", 1.0)) == 0.0
    )
    reason = (
        "v038 stage-0 held-out locomotion/ball-control gate passed"
        if passed
        else "v038 stage-0 held-out locomotion/ball-control gate is not promotion-safe"
    )
    return passed, reason, payload


def require_team_training_gate() -> None:
    """Stop expensive training when the prerequisite evidence is absent or failed."""

    passed, reason, _ = team_training_gate()
    if not passed:
        raise RuntimeError(reason)
