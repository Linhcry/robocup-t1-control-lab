"""在 Booster 官方简化 MuJoCo 模型上无界面验收预训练行走策略。

这个脚本不训练网络，只回答一个关键问题：官方策略本身能否稳定执行前进、横移和
转向。通过后再做 RCSSServerMJ 模型适配，就能把“策略能力”和“接口问题”分开排查。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np


# 允许用户直接运行本文件，而不必先把项目安装成 Python 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locomotion.official_t1_policy import (  # noqa: E402
    DEFAULT_QPOS,
    MotionCommand,
    OfficialT1LocomotionPolicy,
    quat_rotate_inverse_xyzw,
)


# 官方腿部 PD 参数顺序为左腿 6 个关节，再右腿 6 个关节。
LEG_KP = np.asarray([200.0, 200.0, 200.0, 200.0, 50.0, 50.0] * 2, dtype=np.float32)
LEG_KD = np.asarray([5.0, 5.0, 5.0, 5.0, 1.0, 1.0] * 2, dtype=np.float32)


@dataclass
class EpisodeMetrics:
    """单条速度指令的可复核验收结果。"""

    scenario: str
    command_vx: float
    command_vy: float
    command_yaw_rate: float
    duration_s: float
    fell: bool
    min_height_m: float
    delta_x_m: float
    delta_y_m: float
    delta_yaw_rad: float
    mean_vx_mps: float
    mean_vy_mps: float
    mean_yaw_rate_rps: float
    all_23_targets_sent: bool


def quaternion_wxyz_to_yaw(quaternion: np.ndarray) -> float:
    """把 MuJoCo 的 ``w,x,y,z`` 根四元数转为航向角。"""

    w, x, y, z = (float(value) for value in quaternion)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def reset_robot(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """把官方 12 腿关节模型放回统一的初始站姿。"""

    mujoco.mj_resetData(model, data)
    data.qpos[:3] = np.asarray([0.0, 0.0, 0.72], dtype=np.float64)
    data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[7:] = DEFAULT_QPOS[11:]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def run_episode(
    model: mujoco.MjModel,
    policy: OfficialT1LocomotionPolicy,
    scenario: str,
    command: MotionCommand,
    warmup_s: float,
    duration_s: float,
) -> EpisodeMetrics:
    """先稳定站立，再执行一条固定速度指令并统计真实位移。"""

    data = mujoco.MjData(model)
    reset_robot(model, data)
    policy.reset()

    physics_dt = float(model.opt.timestep)
    control_decimation = max(1, round(policy.policy_dt / physics_dt))
    total_steps = round((warmup_s + duration_s) / physics_dt)
    warmup_steps = round(warmup_s / physics_dt)
    leg_targets = DEFAULT_QPOS[11:].copy()
    start_pose: np.ndarray | None = None
    sampled_velocities: list[np.ndarray] = []
    sampled_yaw_rates: list[float] = []
    min_height = float(data.qpos[2])
    fell = False
    target_min = np.full(23, np.inf, dtype=np.float32)
    target_max = np.full(23, -np.inf, dtype=np.float32)

    for physics_step in range(total_steps):
        time_now = physics_step * physics_dt
        active_command = MotionCommand() if physics_step < warmup_steps else command

        if physics_step % control_decimation == 0:
            leg_pos = data.qpos[7:].astype(np.float32)
            leg_vel = data.qvel[6:].astype(np.float32)
            full_pos = DEFAULT_QPOS.copy()
            full_vel = np.zeros(23, dtype=np.float32)
            full_pos[11:] = leg_pos
            full_vel[11:] = leg_vel

            # 官方 XML 的传感器四元数是 wxyz，控制器使用 xyzw，因此显式重排。
            quat_wxyz = data.sensor("orientation").data.astype(np.float32)
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
            projected_gravity = quat_rotate_inverse_xyzw(
                quat_xyzw, np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
            )
            base_ang_vel = data.sensor("angular-velocity").data.astype(np.float32)
            full_targets = policy.infer(
                time_now=time_now,
                joint_pos=full_pos,
                joint_vel=full_vel,
                base_ang_vel=base_ang_vel,
                projected_gravity=projected_gravity,
                command=active_command,
            )
            leg_targets[:] = full_targets[11:]
            target_min = np.minimum(target_min, full_targets)
            target_max = np.maximum(target_max, full_targets)

        leg_pos = data.qpos[7:].astype(np.float32)
        leg_vel = data.qvel[6:].astype(np.float32)
        requested_torque = LEG_KP * (leg_targets - leg_pos) - LEG_KD * leg_vel
        data.ctrl[:] = np.clip(
            requested_torque,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(model, data)

        min_height = min(min_height, float(data.qpos[2]))
        if float(data.qpos[2]) < 0.42 or not np.isfinite(data.qpos).all():
            fell = True
            break

        if physics_step == warmup_steps:
            start_pose = data.qpos[:7].copy()
        if physics_step > warmup_steps + round(0.5 * duration_s / physics_dt):
            sampled_velocities.append(data.qvel[:2].copy())
            sampled_yaw_rates.append(float(data.qvel[5]))

    if start_pose is None:
        start_pose = data.qpos[:7].copy()
    start_yaw = quaternion_wxyz_to_yaw(start_pose[3:7])
    final_yaw = quaternion_wxyz_to_yaw(data.qpos[3:7])
    yaw_delta = float(np.arctan2(np.sin(final_yaw - start_yaw), np.cos(final_yaw - start_yaw)))
    mean_velocity = (
        np.mean(np.asarray(sampled_velocities), axis=0) if sampled_velocities else np.zeros(2)
    )

    # 固定站姿的关节不一定随每条命令变化；这里验证 23 项均被控制器明确赋值且有限。
    targets_are_valid = bool(np.isfinite(target_min).all() and np.isfinite(target_max).all())
    return EpisodeMetrics(
        scenario=scenario,
        command_vx=command.vx,
        command_vy=command.vy,
        command_yaw_rate=command.yaw_rate,
        duration_s=duration_s,
        fell=fell,
        min_height_m=min_height,
        delta_x_m=float(data.qpos[0] - start_pose[0]),
        delta_y_m=float(data.qpos[1] - start_pose[1]),
        delta_yaw_rad=yaw_delta,
        mean_vx_mps=float(mean_velocity[0]),
        mean_vy_mps=float(mean_velocity[1]),
        mean_yaw_rate_rps=float(np.mean(sampled_yaw_rates)) if sampled_yaw_rates else 0.0,
        all_23_targets_sent=targets_are_valid,
    )


def main() -> None:
    """执行三类基本动作并把结果写成 JSON，供后续 RCSS 适配前后对比。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=float, default=2.0, help="每回合先站稳的秒数")
    parser.add_argument("--duration", type=float, default=6.0, help="速度指令持续秒数")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "records" / "v032_official_policy_baseline.json",
        help="验收指标 JSON 输出位置",
    )
    args = parser.parse_args()

    official_root = PROJECT_ROOT.parents[1] / "third_party" / "booster_gym"
    policy_path = official_root / "deploy" / "models" / "T1.pt"
    model_path = official_root / "resources" / "T1" / "T1_locomotion.xml"

    policy = OfficialT1LocomotionPolicy(policy_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = 0.002

    scenarios = {
        "forward": MotionCommand(vx=0.45),
        "lateral": MotionCommand(vy=0.30),
        "turn": MotionCommand(yaw_rate=0.60),
    }
    results = [
        run_episode(model, policy, name, command, args.warmup, args.duration)
        for name, command in scenarios.items()
    ]

    payload = {
        "source_policy": str(policy_path),
        "source_model": str(model_path),
        "joint_interface": "23 targets = 2 head + 8 arms + 1 waist + 12 learned legs",
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
