"""Non-overwriting artifact helpers for v039."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def next_available_path(path: Path) -> Path:
    """Keep historical evidence by adding ``_run_XX`` when a path already exists."""

    if not path.exists():
        return path
    for index in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}_run_{index:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No free artifact name near {path}")


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = next_available_path(path)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination
