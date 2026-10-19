"""2026 中国机器人大赛 Booster T1 的 23 自由度控制合约。

所有主动控制代码只能从这里读取关节顺序、服务器电机名和机械限位，避免训练、
录像、TCP 客户端各自维护一份容易错位的表格。MuJoCo 使用弧度，因此角度在模块
加载时统一从度转换为弧度。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np


@dataclass(frozen=True)
class T1JointSpec:
    """一个关节在规则和 RCSSServerMJ 中的固定定义。"""

    name: str  # 规则文档中的可读名称。
    motor: str  # 服务器动作接口使用的短电机名。
    lower_deg: float  # 最小合法角度。
    upper_deg: float  # 最大合法角度。


# 顺序必须保持：头 2、左臂 4、右臂 4、腰 1、左腿 6、右腿 6。
T1_JOINT_SPECS: tuple[T1JointSpec, ...] = (
    T1JointSpec("Head_yaw", "he1", -90.0, 90.0),
    T1JointSpec("Head_pitch", "he2", -20.0, 70.0),
    T1JointSpec("Left_Shoulder_Pitch", "lae1", -190.0, 70.0),
    T1JointSpec("Left_Shoulder_Roll", "lae2", -100.0, 90.0),
    T1JointSpec("Left_Elbow_Pitch", "lae3", -130.0, 130.0),
    T1JointSpec("Left_Elbow_Yaw", "lae4", -140.0, 0.0),
    T1JointSpec("Right_Shoulder_Pitch", "rae1", -190.0, 70.0),
    T1JointSpec("Right_Shoulder_Roll", "rae2", -90.0, 100.0),
    T1JointSpec("Right_Elbow_Pitch", "rae3", -130.0, 130.0),
    T1JointSpec("Right_Elbow_Yaw", "rae4", 0.0, 140.0),
    T1JointSpec("Waist", "te1", -90.0, 90.0),
    T1JointSpec("Left_Hip_Pitch", "lle1", -103.0, 90.0),
    T1JointSpec("Left_Hip_Roll", "lle2", -11.5, 90.0),
    T1JointSpec("Left_Hip_Yaw", "lle3", -57.3, 57.3),
    T1JointSpec("Left_Knee_Pitch", "lle4", 0.0, 134.0),
    T1JointSpec("Left_Ankle_Pitch", "lle5", -50.0, 20.0),
    T1JointSpec("Left_Ankle_Roll", "lle6", -25.0, 25.0),
    T1JointSpec("Right_Hip_Pitch", "rle1", -103.0, 90.0),
    T1JointSpec("Right_Hip_Roll", "rle2", -90.0, 11.5),
    T1JointSpec("Right_Hip_Yaw", "rle3", -57.3, 57.3),
    T1JointSpec("Right_Knee_Pitch", "rle4", 0.0, 134.0),
    T1JointSpec("Right_Ankle_Pitch", "rle5", -50.0, 20.0),
    T1JointSpec("Right_Ankle_Roll", "rle6", -25.0, 25.0),
)

# 三个数组分别供模型核对、动作发送和限位裁剪使用。
T1_JOINT_NAMES = tuple(spec.name for spec in T1_JOINT_SPECS)
T1_MOTOR_NAMES = tuple(spec.motor for spec in T1_JOINT_SPECS)
T1_LOWER_LIMITS = np.asarray([spec.lower_deg * pi / 180.0 for spec in T1_JOINT_SPECS], dtype=np.float32)
T1_UPPER_LIMITS = np.asarray([spec.upper_deg * pi / 180.0 for spec in T1_JOINT_SPECS], dtype=np.float32)

# RCSSServerMJ robot.xml stores slightly rounded radian ranges.  They are a few
# milliradians tighter than the rule-table degree conversion above, so deployed
# targets must use the intersection of both contracts.
T1_RCSS_MODEL_LOWER_LIMITS = np.asarray(
    [
        -1.57, -0.35, -3.31, -1.74, -2.27, -2.44,
        -3.31, -1.57, -2.27, 0.0, -1.57,
        -1.80, -0.20, -1.00, 0.0, -0.87, -0.44,
        -1.80, -1.57, -1.00, 0.0, -0.87, -0.44,
    ],
    dtype=np.float32,
)
T1_RCSS_MODEL_UPPER_LIMITS = np.asarray(
    [
        1.57, 1.22, 1.22, 1.57, 2.27, 0.0,
        1.22, 1.74, 2.27, 2.44, 1.57,
        1.57, 1.57, 1.00, 2.34, 0.35, 0.44,
        1.57, 0.20, 1.00, 2.34, 0.35, 0.44,
    ],
    dtype=np.float32,
)
T1_SAFE_HARD_LOWER_LIMITS = np.maximum(
    T1_LOWER_LIMITS, T1_RCSS_MODEL_LOWER_LIMITS
).astype(np.float32)
T1_SAFE_HARD_UPPER_LIMITS = np.minimum(
    T1_UPPER_LIMITS, T1_RCSS_MODEL_UPPER_LIMITS
).astype(np.float32)

# One 50 Hz decision may move a target by at most 0.06 rad.  An 0.08 rad
# interior margin plus 25 ms velocity look-ahead starts braking before a soft
# MuJoCo joint constraint is reached.  The actor output itself remains raw and
# learnable; these constants define the shared training/deployment safety layer.
T1_ACTION_DELTA_LIMIT_RAD = 0.06
T1_TARGET_LIMIT_MARGIN_RAD = 0.08
T1_VELOCITY_BRAKE_LOOKAHEAD_S = 0.025


def clamp_t1_targets(targets: np.ndarray) -> np.ndarray:
    """拒绝错误维度，清除非有限值，并按官方规则裁剪 23 维目标角。"""

    values = np.asarray(targets, dtype=np.float32)
    if values.shape != (23,):
        raise ValueError(f"T1 目标必须是 23 维，实际为 {values.shape}")
    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(safe_values, T1_LOWER_LIMITS, T1_UPPER_LIMITS).astype(np.float32)


def filter_t1_dynamic_targets(
    requested_targets: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    *,
    max_delta_rad: float = T1_ACTION_DELTA_LIMIT_RAD,
) -> np.ndarray:
    """Apply the identical 50 Hz target-rate, margin and braking safety layer."""

    requested = np.asarray(requested_targets, dtype=np.float32)
    position = np.asarray(joint_pos, dtype=np.float32)
    velocity = np.asarray(joint_vel, dtype=np.float32)
    for name, values in (
        ("requested_targets", requested),
        ("joint_pos", position),
        ("joint_vel", velocity),
    ):
        if values.shape != (23,):
            raise ValueError(f"{name} must have shape (23,), got {values.shape}")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{name} contains NaN/Inf")
    if not np.isfinite(max_delta_rad) or max_delta_rad <= 0.0:
        raise ValueError("max_delta_rad must be one positive finite scalar")

    bounded_delta = np.clip(
        requested - position,
        -max_delta_rad,
        max_delta_rad,
    )
    bounded_request = position + bounded_delta
    target_lower = T1_SAFE_HARD_LOWER_LIMITS + T1_TARGET_LIMIT_MARGIN_RAD
    target_upper = T1_SAFE_HARD_UPPER_LIMITS - T1_TARGET_LIMIT_MARGIN_RAD
    midpoint = 0.5 * (target_lower + target_upper)
    dynamic_lower = np.minimum(
        target_lower
        + np.maximum(-velocity, 0.0) * T1_VELOCITY_BRAKE_LOOKAHEAD_S,
        midpoint,
    )
    dynamic_upper = np.maximum(
        target_upper
        - np.maximum(velocity, 0.0) * T1_VELOCITY_BRAKE_LOOKAHEAD_S,
        midpoint,
    )
    return np.clip(bounded_request, dynamic_lower, dynamic_upper).astype(np.float32)
