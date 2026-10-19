"""v038 八个高层动作的有限值、站立和 23DoF 冒烟测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v038_high_speed_dribble.curriculum import (  # noqa: E402
    DEFAULT_CONFIG,
    load_config,
    unique_metrics_path,
)
from experiments.v038_high_speed_dribble.high_speed_dribble_env import (  # noqa: E402
    HighSpeedResidualDribbleEnv,
)


def main() -> None:
    config = load_config(DEFAULT_CONFIG)
    env = HighSpeedResidualDribbleEnv(config, 0, seed=38000)
    observation, info = env.reset(seed=38000)
    rewards: list[float] = []
    try:
        for _ in range(8):
            observation, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            rewards.append(float(reward))
            if terminated or truncated:
                break
    finally:
        env.close()
    payload = {
        "version": config["version"],
        "requested_steps": 8,
        "completed_steps": len(rewards),
        "finite_observation": bool(np.isfinite(observation).all()),
        "finite_rewards": bool(np.isfinite(rewards).all()),
        "fell": bool(info["fell"]),
        "minimum_robot_height_m": float(info["minimum_robot_height_m"]),
        "minimum_robot_upright": float(info["minimum_robot_upright"]),
        "all_23_targets_sent": bool(info["all_23_targets_sent"]),
    }
    payload["quality_passed"] = bool(
        payload["completed_steps"] == 8
        and payload["finite_observation"]
        and payload["finite_rewards"]
        and not payload["fell"]
        and payload["all_23_targets_sent"]
    )
    metrics = unique_metrics_path("v038_smoke_8_steps")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(metrics)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
