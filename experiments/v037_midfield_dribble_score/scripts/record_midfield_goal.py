"""加载最终 v037-r2 模型，录制从中场连续控球到官方进球的完整视频。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from experiments.v037_midfield_dribble_score.curriculum import DEFAULT_CONFIG, load_config  # noqa: E402
from experiments.v037_midfield_dribble_score.midfield_dribble_env import ResidualMidfieldDribbleScoreEnv  # noqa: E402
from experiments.v037_midfield_dribble_score.residual_curriculum import latest_residual_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed", type=int, default=37410)
    parser.add_argument("--out", type=Path, default=EXPERIMENT_ROOT / "videos" / "v037_midfield_full_goal.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=5)
    args = parser.parse_args()
    checkpoint = args.model or latest_residual_checkpoint(4)
    if checkpoint is None:
        raise FileNotFoundError("final residual midfield checkpoint is missing")

    config = load_config(args.config)
    model = PPO.load(str(checkpoint), device="cpu")
    env = ResidualMidfieldDribbleScoreEnv(config, 4, seed=args.seed)
    observation, _ = env.reset(seed=args.seed)
    renderer = mujoco.Renderer(env.simulation.mj_model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.2
    camera.azimuth = 40.0
    camera.elevation = -20.0
    render_every = max(1, round(1.0 / (args.video_fps * env.decision_dt)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_reward = 0.0
    final_info: dict[str, object] = {}

    def frame() -> np.ndarray:
        midpoint = 0.58 * np.asarray(env.agent.xpos) + 0.42 * np.asarray(env.simulation.ball.xpos)
        camera.lookat[:] = midpoint
        camera.lookat[2] = 0.45
        renderer.update_scene(env.simulation.mj_data, camera)
        return renderer.render()

    try:
        with imageio.get_writer(args.out, fps=args.video_fps, quality=8, macro_block_size=1) as writer:
            writer.append_data(frame())
            high_step = 0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(action)
                total_reward += float(reward)
                high_step += 1
                if high_step % render_every == 0 or terminated or truncated:
                    writer.append_data(frame())
                if terminated or truncated:
                    break
    finally:
        renderer.close()
        env.close()

    metrics = args.out.with_suffix(".json")
    payload = {
        "model": str(Path(checkpoint).resolve()),
        "seed": args.seed,
        "episode_reward": total_reward,
        "video": str(args.out.resolve()),
        **final_info,
    }
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(metrics.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
