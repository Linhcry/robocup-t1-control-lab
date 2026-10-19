"""Non-overwriting artifact allocation for v043 evidence."""

from __future__ import annotations

from pathlib import Path


def next_numbered_path(folder: Path, prefix: str, suffix: str) -> Path:
    """Return the first unused ``prefix_0001_suffix`` path."""

    folder.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        candidate = folder / f"{prefix}_{index:04d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"artifact namespace exhausted in {folder}")
