"""按 Booster 官方 RoboCup Demo 的 T1 ``Kick`` 逻辑走向足球并触球。

官方 T1 普通踢球行为会根据球的相对方位持续发 crab-walk 速度命令，并在球明显
移动后停止。本脚本把同一逻辑接到 RCSSServerMJ，既生成数值验收，也录制视频。
"""

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
    """放置足球、稳定站立、闭环走向球，并在触球后停止。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "videos" / "v033_rcss_walk_kick.mp4",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "records" / "v033_rcss_walk_kick.json",
    )
    parser.add_argument("--no-video", action="store_true", help="只做数值验收，不渲染 MP4")
    parser.add_argument("--max-approach-seconds", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    official_root = PROJECT_ROOT.parents[1] / "third_party" / "booster_gym"
    controller = RcssT1Controller(official_root / "deploy" / "models" / "T1.pt")
    simulation, agent = make_single_t1_simulation()
    controller.reset_robot(simulation, agent)

    # 球略偏机器人右脚，让正常步态自然触球，而不是从两脚正中穿过。
    simulation.ball.drop_at(1.25, -0.08)
    mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
    initial_ball_position = np.asarray(simulation.ball.xpos, dtype=np.float64).copy()
    simulation_dt = float(simulation.timestep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    renderer = None if args.no_video else mujoco.Renderer(simulation.mj_model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.5
    camera.azimuth = 140.0
    camera.elevation = -17.0

    kicked = False
    fell = False
    policy_time = 0.0
    max_ball_speed = 0.0
    min_height = float(agent.xpos[2])
    min_upright = float(torso_upright(agent))
    completed_approach_steps = 0

    def append_frame(writer: object | None) -> None:
        """录像开启时写入一帧，并让镜头同时照到机器人和球。"""

        if renderer is None or writer is None:
            return
        midpoint = 0.72 * np.asarray(agent.xpos) + 0.28 * np.asarray(simulation.ball.xpos)
        camera.lookat[:] = midpoint
        renderer.update_scene(simulation.mj_data, camera)
        writer.append_data(renderer.render())

    writer_context = (
        imageio.get_writer(args.out, fps=args.fps, quality=8, macro_block_size=1)
        if not args.no_video
        else None
    )

    try:
        writer = writer_context

        # 与官方部署一致，先保持准备姿态 3 秒再启动行走网络。
        for _ in range(round(3.0 / simulation_dt)):
            controller.prepare_step(simulation, agent)
            simulation.step([])
            append_frame(writer)

        for approach_step in range(round(args.max_approach_seconds / simulation_dt)):
            # 把世界坐标中的“球减机器人”转换为机器人自身坐标。
            rotation_world_from_body = np.asarray(agent.xmat, dtype=np.float64).reshape(3, 3)
            ball_vector_world = np.asarray(simulation.ball.xpos) - np.asarray(agent.xpos)
            ball_vector_body = rotation_world_from_body.T @ ball_vector_world
            ball_angle = float(np.arctan2(ball_vector_body[1], ball_vector_body[0]))

            # 官方 Kick 会逐步提升 crab-walk 速度；这里限制在已验收安全的 0.45 m/s。
            requested_speed = min(0.45, 0.25 + 0.04 * approach_step * simulation_dt)
            command_vx = requested_speed * float(np.cos(ball_angle))
            command_vy = float(np.clip(requested_speed * np.sin(ball_angle), -0.20, 0.20))
            command = MotionCommand(
                vx=command_vx,
                vy=command_vy,
                yaw_rate=0.0,
                head_yaw=float(np.clip(ball_angle, -0.8, 0.8)),
                head_pitch=0.30,
            )
            controller.command_step(simulation, agent, command, policy_time)
            simulation.step([])
            policy_time += simulation_dt
            completed_approach_steps = approach_step + 1
            append_frame(writer)

            ball_speed = float(np.linalg.norm(simulation.ball._qvel[:3]))
            max_ball_speed = max(max_ball_speed, ball_speed)
            ball_displacement = float(np.linalg.norm(np.asarray(simulation.ball.xpos[:2]) - initial_ball_position[:2]))
            min_height = min(min_height, float(agent.xpos[2]))
            min_upright = min(min_upright, float(torso_upright(agent)))
            kicked = ball_displacement > 0.18 or ball_speed > 0.55
            fell = float(agent.xpos[2]) < 0.34 or float(torso_upright(agent)) < 0.20
            if kicked or fell:
                break

        # 触球后发零速度，让录像清楚显示球继续运动而机器人恢复站立。
        if not fell:
            for _ in range(round(2.0 / simulation_dt)):
                controller.command_step(simulation, agent, MotionCommand(head_pitch=0.25), policy_time)
                simulation.step([])
                policy_time += simulation_dt
                append_frame(writer)
    finally:
        if writer_context is not None:
            writer_context.close()
        if renderer is not None:
            renderer.close()

    final_ball_position = np.asarray(simulation.ball.xpos, dtype=np.float64).copy()
    payload = {
        "behavior": "Booster T1 official crab-walk kick logic",
        "kicked": kicked,
        "fell": fell,
        "all_23_targets_sent": controller.all_23_targets_sent,
        "approach_seconds": completed_approach_steps * simulation_dt,
        "ball_delta_x_m": float(final_ball_position[0] - initial_ball_position[0]),
        "ball_delta_y_m": float(final_ball_position[1] - initial_ball_position[1]),
        "ball_displacement_m": float(np.linalg.norm(final_ball_position[:2] - initial_ball_position[:2])),
        "max_ball_speed_mps": max_ball_speed,
        "min_robot_height_m": min_height,
        "min_robot_upright": min_upright,
        "video": None if args.no_video else str(args.out),
    }
    args.metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
