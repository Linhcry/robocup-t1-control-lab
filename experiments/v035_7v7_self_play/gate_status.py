"""读取 v034 质量门槛，防止不稳定踢球技能进入 7v7 长训练。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V034_ROOT = PROJECT_ROOT / "experiments" / "v034_dynamic_single_leg_kick"


def latest_v034_metrics() -> tuple[Path | None, dict[str, Any] | None]:
    """返回编号最大的 v034 训练指标；没有指标时返回两个 ``None``。"""

    paths = sorted(V034_ROOT.joinpath("metrics").glob("v034_policy_iter_*_training.json"))
    if not paths:
        return None, None
    path = paths[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))


def v035_training_allowed() -> tuple[bool, str]:
    """只有最新 v034 指标明确通过时，才允许启动下一技能长训练。"""

    path, payload = latest_v034_metrics()
    if path is None or payload is None:
        return False, "没有找到 v034 固定种子评估/训练指标"
    if not bool(payload.get("quality_passed", False)):
        return False, f"{path.name} 的 quality_passed=false"
    return True, f"{path.name} 已通过"


def require_v035_training_gate() -> None:
    """训练脚本共同调用的硬门槛；比赛冒烟和只读评估不受影响。"""

    allowed, reason = v035_training_allowed()
    if not allowed:
        raise RuntimeError(
            "v035 长训练尚未获准："
            + reason
            + "。先让 v034 达到跌倒率、球速、方向和踢后恢复门槛；"
            + "当前仍可运行 7v7 冒烟、启发式十分钟比赛和生成演示数据。"
        )

