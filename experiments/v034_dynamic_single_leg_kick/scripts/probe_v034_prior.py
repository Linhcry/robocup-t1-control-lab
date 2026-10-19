"""Print a compact zero-residual v034 trajectory for kick-prior tuning."""

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
from kick_training import DEFAULT_CONFIG, load_config  # noqa: E402


def main() -> None:
    env = DynamicSingleLegKickEnv.from_config(load_config(DEFAULT_CONFIG), seed=34010)
    env.reset(seed=34010)
    rows = []
    try:
        for index in range(env.max_episode_steps):
            _, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
            if index % 2 == 0 or terminated or truncated:
                right_foot = env._foot_position(env.right_foot_geom_id)
                left_foot = env._foot_position(env.left_foot_geom_id)
                rows.append(
                    {
                        "step": index + 1,
                        "phase": info["phase"],
                        "reward": reward,
                        "height": float(env.agent.xpos[2]),
                        "upright": float(env.agent.xmat[8]),
                        "left_foot": left_foot.tolist(),
                        "right_foot": right_foot.tolist(),
                        "ball": np.asarray(env.sim.ball.xpos).tolist(),
                        "right_foot_ball_distance": float(np.linalg.norm(np.asarray(env.sim.ball.xpos) - right_foot)),
                        "right_leg_targets": env.last_targets[17:23].tolist(),
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
            if terminated or truncated:
                break
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()

