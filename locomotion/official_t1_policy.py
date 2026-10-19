"""把 Booster 官方 12 腿关节策略扩展成比赛模型需要的 23 关节目标。

这份文件是当前项目最重要的“机器人控制器”之一：

1. 输入机器人姿态、角速度、23 个关节状态以及期望速度；
2. 调用 Booster 官方预训练策略计算两条腿的 12 个关节目标；
3. 用独立的视线、手臂和腰部控制补齐其余 11 个关节；
4. 返回严格按照中国机器人大赛规则排列的 23 维目标角。

注意：官方开源 ``T1.pt`` 是行走底座，只学习腿部 12 自由度。把头部也塞进
行走网络并不会自动变好；足球机器人通常把头部留给看球控制器。这里仍会在每个
控制周期明确生成全部 23 个目标，因此仿真接口没有遗漏任何自由度。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray

from locomotion.t1_spec import T1_LOWER_LIMITS, T1_UPPER_LIMITS


# 该顺序与比赛规则、RCSSServerMJ 的 T1 电机顺序和项目 ``src/t1_spec.py`` 一致。
FULL_JOINT_ORDER: Final[tuple[str, ...]] = (
    "Head_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)

# 官方部署配置里的默认站姿；前 11 项是头、双臂和腰，后 12 项是双腿。
DEFAULT_QPOS: Final[NDArray[np.float32]] = np.asarray(
    [
        0.0,
        0.0,
        0.2,
        -1.35,
        0.0,
        -0.5,
        0.2,
        1.35,
        0.0,
        0.5,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.25,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.25,
        0.0,
    ],
    dtype=np.float32,
)

def quat_rotate_inverse_xyzw(
    quat_xyzw: NDArray[np.float32], vector_world: NDArray[np.float32]
) -> NDArray[np.float32]:
    """把世界坐标向量旋转到机身坐标，输入四元数顺序为 ``x,y,z,w``。"""

    quat_vector = quat_xyzw[:3]
    quat_w = float(quat_xyzw[3])
    term_a = vector_world * (2.0 * quat_w**2 - 1.0)
    term_b = np.cross(quat_vector, vector_world) * (2.0 * quat_w)
    term_c = quat_vector * (2.0 * float(np.dot(quat_vector, vector_world)))
    return np.asarray(term_a - term_b + term_c, dtype=np.float32)


@dataclass(frozen=True)
class MotionCommand:
    """机器人机身坐标系中的速度指令。"""

    vx: float = 0.0  # 前后速度，正数表示向前，单位 m/s。
    vy: float = 0.0  # 左右速度，正数表示向左，单位 m/s。
    yaw_rate: float = 0.0  # 绕竖直轴的转向角速度，单位 rad/s。
    head_yaw: float = 0.0  # 看球控制器给出的头部水平角。
    head_pitch: float = 0.0  # 看球控制器给出的头部俯仰角。


class OfficialT1LocomotionPolicy:
    """Booster 官方腿部策略与 23 自由度比赛控制层的组合。"""

    policy_dt: Final[float] = 0.02  # 官方控制周期：2 ms 物理步长 × 10 次降采样。
    gait_frequency: Final[float] = 1.0  # 官方部署模型采用 1 Hz 相位时钟。

    def __init__(self, policy_path: str | Path, device: str = "cpu") -> None:
        # TorchScript 文件已经包含网络结构，加载后无需重新定义训练网络。
        self.policy_path = Path(policy_path).expanduser().resolve()
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"找不到官方 T1 策略：{self.policy_path}")
        self.device = torch.device(device)
        self.policy = torch.jit.load(str(self.policy_path), map_location=self.device)
        self.policy.eval()

        # 官方网络的输入为 47 维观测，输出为两条腿的 12 维目标偏移。
        self.observation = np.zeros(47, dtype=np.float32)
        self.last_leg_action = np.zeros(12, dtype=np.float32)
        self.smoothed_command = np.zeros(3, dtype=np.float32)
        self.last_targets = DEFAULT_QPOS.copy()

    def reset(self) -> None:
        """在每个新回合开始时清空网络记忆量和指令平滑器。"""

        self.observation.fill(0.0)
        self.last_leg_action.fill(0.0)
        self.smoothed_command.fill(0.0)
        self.last_targets[:] = DEFAULT_QPOS

    def _smooth_velocity_command(self, command: MotionCommand) -> bool:
        """限制每周期的速度突变，避免一步内给机器人施加不现实的急转指令。"""

        requested = np.asarray([command.vx, command.vy, command.yaw_rate], dtype=np.float32)
        delta = np.clip(requested - self.smoothed_command, -self.policy_dt, self.policy_dt)
        self.smoothed_command += delta
        return bool(np.linalg.norm(self.smoothed_command) >= 1.0e-5)

    def infer(
        self,
        time_now: float,
        joint_pos: NDArray[np.float32],
        joint_vel: NDArray[np.float32],
        base_ang_vel: NDArray[np.float32],
        projected_gravity: NDArray[np.float32],
        command: MotionCommand,
    ) -> NDArray[np.float32]:
        """根据当前状态和速度指令计算下一周期的 23 维关节目标角。"""

        # 尺寸断言会立即暴露“少关节、错顺序”的适配错误，不能静默截断。
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_vel = np.asarray(joint_vel, dtype=np.float32)
        if joint_pos.shape != (23,) or joint_vel.shape != (23,):
            raise ValueError(f"关节状态必须是 23 维，实际为 {joint_pos.shape} / {joint_vel.shape}")

        moving = self._smooth_velocity_command(command)
        phase = np.fmod(time_now * self.gait_frequency, 1.0) if moving else 0.0

        # 0:3 是机身坐标系中的重力方向，用于网络判断机器人是否倾斜。
        self.observation[0:3] = np.asarray(projected_gravity, dtype=np.float32)
        # 3:6 是机身角速度，提供转动和失衡趋势。
        self.observation[3:6] = np.asarray(base_ang_vel, dtype=np.float32)
        # 6:9 是使用者要求的前进、横移和转向速度。
        self.observation[6:9] = self.smoothed_command if moving else 0.0
        # 9:11 是步态相位时钟；静止时归零，避免原地无意义踏步。
        self.observation[9] = np.cos(2.0 * np.pi * phase) if moving else 0.0
        self.observation[10] = np.sin(2.0 * np.pi * phase) if moving else 0.0
        # 11:23 是腿部相对默认站姿的 12 个关节角。
        self.observation[11:23] = joint_pos[11:] - DEFAULT_QPOS[11:]
        # 23:35 是腿部关节速度；官方训练时按 0.1 缩放。
        self.observation[23:35] = 0.1 * joint_vel[11:]
        # 35:47 是上一周期动作，使策略能抑制动作跳变。
        self.observation[35:47] = self.last_leg_action

        # 推理阶段不需要梯度，可降低显存和运行开销。
        with torch.inference_mode():
            tensor_obs = torch.from_numpy(self.observation).unsqueeze(0).to(self.device)
            leg_action = self.policy(tensor_obs).squeeze(0).cpu().numpy().astype(np.float32)
        self.last_leg_action[:] = np.clip(leg_action, -1.0, 1.0)

        # 先恢复完整默认姿态，再把策略输出加到 12 个腿关节上。
        targets = DEFAULT_QPOS.copy()
        targets[11:] += self.last_leg_action

        # 头部由看球方向直接控制；这两个自由度不应被行走网络随机摆动。
        targets[0] = command.head_yaw
        targets[1] = command.head_pitch

        # 双臂做与腿相反的轻量摆臂，既明确调用上肢关节，也帮助抵消躯干角动量。
        speed_scale = float(np.clip(np.linalg.norm(self.smoothed_command[:2]), 0.0, 1.0))
        arm_swing = 0.14 * speed_scale * np.sin(2.0 * np.pi * phase)
        targets[2] += arm_swing
        targets[6] -= arm_swing
        targets[4] -= 0.35 * arm_swing
        targets[8] += 0.35 * arm_swing

        # 腰部只承担小幅转向配合，主要航向仍由两条腿完成。
        targets[10] = 0.08 * float(self.smoothed_command[2])

        # 最后依据 2026 比赛规则裁剪，保证所有目标都处在合法运动空间内。
        targets = np.clip(targets, T1_LOWER_LIMITS, T1_UPPER_LIMITS)
        if not np.isfinite(targets).all():
            raise FloatingPointError("策略产生了 NaN 或 Inf 关节目标，已拒绝发送到仿真器")
        self.last_targets[:] = targets
        return targets.copy()
