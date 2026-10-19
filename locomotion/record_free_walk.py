"""录制完整 RCSS T1 连续执行前进、横移、转向和停止的视频。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locomotion.official_t1_policy import MotionCommand  # noqa: E402
from locomotion.rcss_t1_controller import RcssT1Controller  # noqa: E402
from locomotion.rcss_simulation import make_single_t1_simulation, torso_upright  # noqa: E402


def main() -> None:
    """连续执行安全速度包络，并把画面和每段位移一起保存。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "videos" / "v032_rcss_free_walk.mp4",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "records" / "v032_rcss_free_walk_video.json",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    official_root = PROJECT_ROOT.parents[1] / "third_party" / "booster_gym"
    controller = RcssT1Controller(official_root / "deploy" / "models" / "T1.pt")
    simulation, agent = make_single_t1_simulation()
    controller.reset_robot(simulation, agent)
    simulation_dt = float(simulation.timestep)

    # 每一段都使用已通过独立验收的安全速度，最后回到静止指令。
    schedule = [
        ("前进", 5.0, MotionCommand(vx=0.40)),
        ("横移", 4.0, MotionCommand(vy=0.20)),
        ("转向", 6.0, MotionCommand(yaw_rate=0.15)),
        ("停止", 2.0, MotionCommand()),
    ]
    prepare_seconds = 3.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(simulation.mj_model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.4
    camera.azimuth = 135.0
    camera.elevation = -16.0

    segment_metrics: list[dict[str, float | str | bool]] = []
    fell = False
    policy_time = 0.0

    def append_frame(writer: imageio.core.format.Writer) -> None:
        """让镜头跟随躯干，并写入一帧 RGB 图像。"""

        camera.lookat[:] = np.asarray(agent.xpos, dtype=np.float64)
        renderer.update_scene(simulation.mj_data, camera)
        writer.append_data(renderer.render())

    try:
        with imageio.get_writer(args.out, fps=args.fps, quality=8, macro_block_size=1) as writer:
            # 准备阶段也录入视频，便于看出机器人不是被瞬间“传送”到行走姿态。
            for _ in range(round(prepare_seconds / simulation_dt)):
                controller.prepare_step(simulation, agent)
                simulation.step([])
                append_frame(writer)

            for name, seconds, command in schedule:
                start = np.asarray(agent.xpos, dtype=np.float64).copy()
                min_height = float(agent.xpos[2])
                min_upright = float(torso_upright(agent))
                completed_steps = 0

                for _ in range(round(seconds / simulation_dt)):
                    controller.command_step(simulation, agent, command, policy_time)
                    simulation.step([])
                    policy_time += simulation_dt
                    completed_steps += 1
                    append_frame(writer)

                    min_height = min(min_height, float(agent.xpos[2]))
                    min_upright = min(min_upright, float(torso_upright(agent)))
                    if float(agent.xpos[2]) < 0.34 or float(torso_upright(agent)) < 0.20:
                        fell = True
                        break

                segment_metrics.append(
                    {
                        "segment": name,
                        "requested_seconds": seconds,
                        "completed_seconds": completed_steps * simulation_dt,
                        "delta_x_m": float(agent.xpos[0] - start[0]),
                        "delta_y_m": float(agent.xpos[1] - start[1]),
                        "min_height_m": min_height,
                        "min_upright": min_upright,
                        "fell": fell,
                    }
                )
                if fell:
                    break
    finally:
        renderer.close()

    payload = {
        "video": str(args.out),
        "controller": "locomotion/rcss_t1_controller.py",
        "all_23_targets_sent": controller.all_23_targets_sent,
        "fell": fell,
        "segments": segment_metrics,
    }
    args.metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
