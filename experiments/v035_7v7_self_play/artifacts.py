"""为 v035 分配只增不覆盖的实验产物路径。"""

from __future__ import annotations

from pathlib import Path


def next_numbered_path(folder: Path, prefix: str, suffix: str) -> Path:
    """返回 ``prefix_0001_suffix`` 形式的第一个未使用路径。"""

    folder.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        candidate = folder / f"{prefix}_{index:04d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"{folder} 的实验编号已经用完")

