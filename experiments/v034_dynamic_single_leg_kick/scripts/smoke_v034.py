"""Run exactly eight finite v034 control decisions before any PPO training."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for path in (PROJECT_ROOT, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_kick_env import DynamicSingleLegKickEnv  # noqa: E402
from kick_training import DEFAULT_CONFIG, load_config, unique_metrics_path  # noqa: E402
from locomotion.t1_spec import T1_LOWER_LIMITS, T1_UPPER_LIMITS  # noqa: E402


def main() -> None:
    config = load_config(DEFAULT_CONFIG)
    env = DynamicSingleLegKickEnv.from_config(config, seed=34000)
    observation, _ = env.reset(seed=34000)
    rows = []
    try:
        for step in range(8):
            observation, reward, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            if terminated or truncated:
                raise AssertionError(f"v034 smoke ended at step {step + 1}: {info}")
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                raise AssertionError("v034 smoke produced NaN/Inf")
            if not (
                np.all(env.last_targets >= T1_LOWER_LIMITS - 1.0e-6)
                and np.all(env.last_targets <= T1_UPPER_LIMITS + 1.0e-6)
            ):
                raise AssertionError("v034 smoke escaped T1 joint limits")
            rows.append({"step": step + 1, "reward": float(reward), **info})

        if not env.controller.all_23_targets_sent:
            raise AssertionError("v034 smoke did not command all 23 joints")
        output = unique_metrics_path("v034_smoke_8_steps")
        payload = {
            "status": "PASS",
            "steps": 8,
            "finite_observations": True,
            "finite_rewards": True,
            "targets_inside_joint_limits": True,
            "all_23_targets_sent": True,
            "final": rows[-1],
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**payload, "metrics": str(output)}, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()

