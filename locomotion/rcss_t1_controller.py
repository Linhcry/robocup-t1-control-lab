"""RCSSServerMJ 中 Booster T1 的正式 23 自由度运动控制器。

如果你只想学习“机器人到底怎样被控制”，优先阅读本文件：

``速度指令 -> 官方神经网络 -> 23 维目标角 -> 目标滤波 -> PD/扭矩控制 -> MuJoCo``

神经网络本身位于 ``official_t1_policy.py``；本文件负责比赛模型适配、安全限制和
真正向 23 个电机发命令。训练、评估和录像都应调用这里，避免不同版本各写一套控制。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from locomotion.official_t1_policy import MotionCommand, OfficialT1LocomotionPolicy  # noqa: E402
from locomotion.t1_spec import (  # noqa: E402
    T1_MOTOR_NAMES,
    clamp_t1_targets,
    filter_t1_dynamic_targets,
)


# 官方普通行走模式的逐关节比例增益 Kp。
COMMON_KP = np.asarray(
    [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 200, 200, 200, 200, 200, 50, 50, 200, 200, 200, 200, 50, 50],
    dtype=np.float32,
)

# 官方普通行走模式的逐关节微分增益 Kd。
COMMON_KD = np.asarray(
    [0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5, 5, 5, 5, 5, 3, 3, 5, 5, 5, 5, 3, 3],
    dtype=np.float32,
)

# 电机扭矩上限用于保护并联机构，单位 N·m。
TORQUE_LIMIT = np.asarray(
    [7, 7, 10, 10, 10, 10, 10, 10, 10, 10, 30, 60, 25, 30, 60, 24, 15, 60, 25, 30, 60, 24, 15],
    dtype=np.float32,
)

# 启动强化学习步态前，机器人先进入这个较柔和的准备姿态。
PREPARE_QPOS = np.asarray(
    [0, 0, 0.25, -1.4, 0, -0.5, 0.25, 1.4, 0, 0.5, 0, -0.1, 0, 0, 0.2, -0.1, 0, -0.1, 0, 0, 0.2, -0.1, 0],
    dtype=np.float32,
)

# 准备姿态使用更强的腿部刚度，确保切换前双脚已经稳定承重。
PREPARE_KP = np.asarray(
    [5, 5, 40, 50, 20, 10, 40, 50, 20, 10, 100, 350, 350, 180, 350, 450, 450, 350, 350, 180, 350, 450, 450],
    dtype=np.float32,
)

# 准备阶段阻尼抑制突然落地和关节来回振荡。
PREPARE_KD = np.asarray(
    [0.1, 0.1, 0.5, 1.5, 0.2, 0.2, 0.5, 1.5, 0.2, 0.2, 5, 7.5, 7.5, 3, 5.5, 0.5, 0.5, 7.5, 7.5, 3, 5.5, 0.5, 0.5],
    dtype=np.float32,
)

# 两个踝关节的 pitch/roll 在 T1 中属于串并联机构，官方部署要求单独处理。
PARALLEL_MECH_INDEXES = frozenset((15, 16, 21, 22))


def read_joint_state(agent: object) -> tuple[np.ndarray, np.ndarray]:
    """按照规则顺序读取 23 个关节的位置和速度。"""

    joint_pos = np.asarray([float(joint.qpos[0]) for joint in agent._joints], dtype=np.float32)
    joint_vel = np.asarray([float(joint.qvel[0]) for joint in agent._joints], dtype=np.float32)
    if joint_pos.shape != (23,) or joint_vel.shape != (23,):
        raise RuntimeError(f"比赛模型关节数量错误：{joint_pos.shape} / {joint_vel.shape}")
    return joint_pos, joint_vel


class RcssT1Controller:
    """把速度命令稳定、安全地转换成 RCSSServerMJ 的 23 路电机命令。"""

    def __init__(self, policy_path: str | Path, *, policy_device: str = "cpu") -> None:
        # 预训练策略负责产生双腿动作，本类负责完整机器人和执行器细节。
        self.policy = OfficialT1LocomotionPolicy(policy_path, device=policy_device)
        self.filtered_targets = PREPARE_QPOS.copy()
        self.target_was_sent = np.zeros(23, dtype=bool)

    def reset_robot(self, simulation: object, agent: object) -> None:
        """重置物理状态并直接放入官方准备姿态。"""

        agent.drop_at(0.0, 0.0, 0.0)
        agent.init_joints(pos=clamp_t1_targets(PREPARE_QPOS).astype(np.float64))
        simulation.mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
        self.policy.reset()
        self.filtered_targets[:] = PREPARE_QPOS
        self.target_was_sent.fill(False)

    def prepare_step(self, simulation: object, agent: object) -> None:
        """发送一次准备姿态控制；调用者随后执行一次 ``simulation.step``。"""

        prefix = agent.agent_id.prefix
        for index, (motor, target) in enumerate(zip(T1_MOTOR_NAMES, PREPARE_QPOS, strict=True)):
            simulation.ctrl_motor(
                prefix + motor,
                float(target),
                0.0,
                float(PREPARE_KP[index]),
                float(PREPARE_KD[index]),
                0.0,
            )

    def _send_targets(
        self,
        simulation: object,
        agent: object,
        targets: np.ndarray,
        *,
        joint_pos: np.ndarray | None = None,
        joint_vel: np.ndarray | None = None,
        dynamic_safety: bool = True,
    ) -> np.ndarray:
        """Send one finite, range-clamped 23-DoF target through the official mapping."""

        requested = np.asarray(targets, dtype=np.float32)
        if requested.shape != (23,):
            raise ValueError(f"T1 target must have shape (23,), got {requested.shape}")
        if not np.isfinite(requested).all():
            raise FloatingPointError("Refusing to send NaN/Inf T1 targets")
        if joint_pos is None:
            joint_pos, measured_vel = read_joint_state(agent)
            if joint_vel is None:
                joint_vel = measured_vel
        elif joint_vel is None:
            _, joint_vel = read_joint_state(agent)
        safe_targets = (
            filter_t1_dynamic_targets(requested, joint_pos, joint_vel)
            if dynamic_safety
            else clamp_t1_targets(requested)
        )

        prefix = agent.agent_id.prefix
        for index, (motor, target) in enumerate(zip(T1_MOTOR_NAMES, safe_targets, strict=True)):
            if index in PARALLEL_MECH_INDEXES:
                direct_torque = float(
                    np.clip(
                        (target - joint_pos[index]) * COMMON_KP[index],
                        -TORQUE_LIMIT[index],
                        TORQUE_LIMIT[index],
                    )
                )
                simulation.ctrl_motor(
                    prefix + motor,
                    float(joint_pos[index]),
                    0.0,
                    0.0,
                    float(COMMON_KD[index]),
                    direct_torque,
                )
            else:
                simulation.ctrl_motor(
                    prefix + motor,
                    float(target),
                    0.0,
                    float(COMMON_KP[index]),
                    float(COMMON_KD[index]),
                    0.0,
                )
            self.target_was_sent[index] = True
        return safe_targets.copy()

    def target_step(
        self,
        simulation: object,
        agent: object,
        targets: np.ndarray,
        *,
        retention: float = 0.0,
    ) -> np.ndarray:
        """Send learned-skill targets without bypassing finite checks or joint limits."""

        requested = np.asarray(targets, dtype=np.float32)
        if requested.shape != (23,):
            raise ValueError(f"T1 target must have shape (23,), got {requested.shape}")
        if not np.isfinite(requested).all():
            raise FloatingPointError("Refusing to send NaN/Inf T1 targets")
        if not 0.0 <= retention < 1.0:
            raise ValueError(f"retention must be in [0, 1), got {retention}")

        joint_pos, joint_vel = read_joint_state(agent)
        safe_targets = filter_t1_dynamic_targets(
            requested, joint_pos, joint_vel
        )
        self.filtered_targets *= retention
        self.filtered_targets += (1.0 - retention) * safe_targets
        self.filtered_targets[:] = clamp_t1_targets(self.filtered_targets)
        return self._send_targets(
            simulation,
            agent,
            self.filtered_targets,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

    def command_step(
        self,
        simulation: object,
        agent: object,
        command: MotionCommand,
        policy_time: float,
    ) -> np.ndarray:
        """计算并发送一次行走控制，返回本周期实际发送的 23 维目标角。"""

        joint_pos, joint_vel = read_joint_state(agent)

        # xmat 是“机身坐标旋转到世界坐标”的矩阵，转置后即可获得机身系传感量。
        rotation_world_from_body = np.asarray(agent.xmat, dtype=np.float32).reshape(3, 3)
        projected_gravity = rotation_world_from_body.T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        # MuJoCo 自由关节 qvel 的后三项已经是躯干切空间中的角速度；如果再乘一次
        # 世界到机身旋转矩阵，朝向 pi 的球员会把 roll/pitch 反馈反号并迅速跌倒。
        body_ang_vel = np.asarray(agent._qvel[3:6], dtype=np.float32)

        # 神经网络根据当前平衡状态和 vx/vy/yaw 指令生成下一组完整目标。
        raw_targets = self.policy.infer(
            time_now=policy_time,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            base_ang_vel=body_ang_vel,
            projected_gravity=projected_gravity,
            command=command,
        )

        # 官方发布线程每 2 ms 做一次 0.8/0.2 低通；这里折算到 RCSS 的代理周期。
        publish_substeps = max(1, round(float(simulation.timestep) / 0.002))
        retention = 0.8**publish_substeps
        self.filtered_targets *= retention
        self.filtered_targets += (1.0 - retention) * raw_targets
        self.filtered_targets[:] = clamp_t1_targets(self.filtered_targets)

        # 普通关节使用 PD 位置控制，并联踝关节把位置误差转换为限幅扭矩。
        return self._send_targets(
            simulation,
            agent,
            self.filtered_targets,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            # The official gait needs the full mechanically clamped position
            # target. Limiting every target to 0.06 rad from the measured
            # joint also limits Kp*error to about 12 N*m on leg joints and
            # removes the corrective authority that keeps the robot upright.
            # RCSS native actuators still enforce their physical force caps.
            dynamic_safety=False,
        )

    @property
    def all_23_targets_sent(self) -> bool:
        """最近一次重置后是否已调用过全部 23 个电机。"""

        return bool(self.target_was_sent.all())
