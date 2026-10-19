"""Append-only JSON evidence helpers for v048."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = EXPERIMENT_ROOT / "metrics"


def write_new_json(stem: str, payload: dict[str, Any]) -> Path:
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10000):
        path = METRICS_ROOT / f"{stem}_run_{run:02d}.json"
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"no free append-only evidence filename for {stem}")

