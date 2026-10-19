"""逐项复刻 Booster 官方部署控制器，在 RCSSServerMJ 完整 T1 上验收。

本脚本与早期适配实验的根本差别是：

* 先进入官方 ``prepare`` 姿态，再切换强化学习步态；
* 使用官方逐关节刚度、阻尼和扭矩上限；
* 对四个并联机构关节使用“位置误差转直接扭矩”的特殊处理；
* 按官方发布线程等效地做目标低通滤波；
* 官方网络只改变 12 个腿关节，其余 11 个关节保持安全默认姿态。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rcsssmj.games.soccer.sim.soccer_sim import SoccerSimulation  # noqa: E402

from locomotion.official_t1_policy import MotionCommand  # noqa: E402
from locomotion.rcss_t1_controller import PARALLEL_MECH_INDEXES, RcssT1Controller  # noqa: E402
from locomotion.rcss_simulation import make_single_t1_simulation, torso_upright  # noqa: E402


@dataclass
class DeployMetrics:
    """官方部署方式在完整比赛模型上的一回合结果。"""

    scenario: str
    command_vx: float
    command_vy: float
    command_yaw_rate: float
    survived_s: float
    fell: bool
    min_height_m: float
    min_upright: float
    delta_x_m: float
    delta_y_m: float
    delta_yaw_rad: float
    mean_world_vx_mps: float
    mean_world_vy_mps: float
    mean_body_yaw_rate_rps: float
    all_23_targets_sent: bool


def yaw_from_wxyz(quaternion: np.ndarray) -> float:
    """把根节点 wxyz 四元数转换成航向角。"""

    w, x, y, z = (float(item) for item in quaternion)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def run_episode(
    simulation: SoccerSimulation,
    agent: object,
    controller: RcssT1Controller,
    name: str,
    command: MotionCommand,
    prepare_s: float,
    duration_s: float,
) -> DeployMetrics:
    """执行官方准备态和行走态，记录真实位移及是否跌倒。"""

    controller.reset_robot(simulation, agent)

    simulation_dt = float(simulation.timestep)
    prepare_steps = round(prepare_s / simulation_dt)
    rl_steps = round(duration_s / simulation_dt)
    min_height = float(agent.xpos[2])
    min_upright = float(torso_upright(agent))
    fell = False
    survived_steps = 0

    # 官方实机由操作者确认准备姿态稳定后才启动 RL 步态，因此两阶段分开运行。
    for step in range(prepare_steps):
        controller.prepare_step(simulation, agent)
        simulation.step([])
        survived_steps += 1
        min_height = min(min_height, float(agent.xpos[2]))
        min_upright = min(min_upright, float(torso_upright(agent)))
        if float(agent.xpos[2]) < 0.34 or float(torso_upright(agent)) < 0.20:
            fell = True
            break

    start_position = np.asarray(agent.xpos, dtype=np.float64).copy()
    start_yaw = yaw_from_wxyz(np.asarray(agent.xquat))
    velocity_samples: list[np.ndarray] = []
    yaw_rate_samples: list[float] = []

    if not fell:
        for rl_step in range(rl_steps):
            time_now = rl_step * simulation_dt
            controller.command_step(simulation, agent, command, time_now)
            simulation.step([])
            survived_steps += 1

            height = float(agent.xpos[2])
            upright = float(torso_upright(agent))
            min_height = min(min_height, height)
            min_upright = min(min_upright, upright)
            if height < 0.34 or upright < 0.20 or not np.isfinite(agent.xpos).all():
                fell = True
                break

            if rl_step > rl_steps // 2:
                velocity_samples.append(np.asarray(agent._qvel[:2], dtype=np.float64).copy())
                rotation_world_from_body = np.asarray(agent.xmat, dtype=np.float32).reshape(3, 3)
                yaw_rate_samples.append(float((rotation_world_from_body.T @ np.asarray(agent._qvel[3:6]))[2]))

    final_yaw = yaw_from_wxyz(np.asarray(agent.xquat))
    yaw_delta = float(np.arctan2(np.sin(final_yaw - start_yaw), np.cos(final_yaw - start_yaw)))
    mean_velocity = np.mean(np.asarray(velocity_samples), axis=0) if velocity_samples else np.zeros(2)
    return DeployMetrics(
        scenario=name,
        command_vx=command.vx,
        command_vy=command.vy,
        command_yaw_rate=command.yaw_rate,
        survived_s=survived_steps * simulation_dt,
        fell=fell,
        min_height_m=min_height,
        min_upright=min_upright,
        delta_x_m=float(agent.xpos[0] - start_position[0]),
        delta_y_m=float(agent.xpos[1] - start_position[1]),
        delta_yaw_rad=yaw_delta,
        mean_world_vx_mps=float(mean_velocity[0]),
        mean_world_vy_mps=float(mean_velocity[1]),
        mean_body_yaw_rate_rps=float(np.mean(yaw_rate_samples)) if yaw_rate_samples else 0.0,
        all_23_targets_sent=controller.all_23_targets_sent,
    )


def main() -> None:
    """运行站立、前进、横移、转向四项官方部署验收。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["stand", "forward", "lateral", "turn"],
        default=["stand", "forward", "lateral", "turn"],
    )
    parser.add_argument("--forward-speed", type=float, default=0.45)
    parser.add_argument("--lateral-speed", type=float, default=0.30)
    parser.add_argument("--yaw-rate", type=float, default=0.60)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "records" / "v032_rcss_exact_official_deploy.json",
    )
    args = parser.parse_args()

    official_root = PROJECT_ROOT.parents[1] / "third_party" / "booster_gym"
    controller = RcssT1Controller(official_root / "deploy" / "models" / "T1.pt")
    simulation, agent = make_single_t1_simulation()
    commands = {
        "stand": MotionCommand(),
        "forward": MotionCommand(vx=args.forward_speed),
        "lateral": MotionCommand(vy=args.lateral_speed),
        "turn": MotionCommand(yaw_rate=args.yaw_rate),
    }
    results = [
        run_episode(simulation, agent, controller, name, commands[name], args.prepare, args.duration)
        for name in args.scenarios
    ]
    payload = {
        "method": "exact Booster deploy sequence adapted to RCSSServerMJ",
        "sim_dt": float(simulation.timestep),
        "parallel_mechanism_indexes": sorted(PARALLEL_MECH_INDEXES),
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
