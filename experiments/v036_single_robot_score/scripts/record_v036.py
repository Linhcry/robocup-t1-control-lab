"""加载 v036 模型，录制一场单机器人追球进球视频并保存数值指标。"""

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

from experiments.v036_single_robot_score.single_robot_score_env import SingleRobotScoreEnv  # noqa: E402
from experiments.v036_single_robot_score.training import DEFAULT_CONFIG, latest_checkpoint, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed", type=int, default=36010)
    parser.add_argument("--out", type=Path, default=EXPERIMENT_ROOT / "videos" / "v036_trained_goal.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    checkpoint = args.model or latest_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError("no v036 checkpoint exists; train v036 first")
    config = load_config(args.config)
    policy = PPO.load(str(checkpoint), device="cpu")
    env = SingleRobotScoreEnv(config, seed=args.seed)
    observation, _ = env.reset(seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out.with_suffix(".json")
    renderer = mujoco.Renderer(env.simulation.mj_model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    # 从球场内侧斜看球门，避免镜头穿过球网造成遮挡。
    camera.azimuth = 40.0
    camera.elevation = -18.0
    total_reward = 0.0
    final_info: dict[str, object] = {}

    def frame() -> np.ndarray:
        midpoint = 0.62 * np.asarray(env.agent.xpos) + 0.38 * np.asarray(env.simulation.ball.xpos)
        camera.lookat[:] = midpoint
        camera.lookat[2] = 0.45
        renderer.update_scene(env.simulation.mj_data, camera)
        return renderer.render()

    try:
        with imageio.get_writer(args.out, fps=round(1.0 / env.decision_dt), quality=8, macro_block_size=1) as writer:
            writer.append_data(frame())
            while True:
                action, _ = policy.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(action)
                total_reward += float(reward)
                writer.append_data(frame())
                if terminated or truncated:
                    break
    finally:
        renderer.close()
        env.close()

    payload = {
        "model": str(Path(checkpoint).resolve()),
        "seed": args.seed,
        "episode_reward": total_reward,
        "video": str(args.out.resolve()),
        **final_info,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(metrics_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
