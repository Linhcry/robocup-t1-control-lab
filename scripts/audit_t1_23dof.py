"""快速验证 2026 Booster T1 的 23 自由度、执行器映射和机械限位。"""

from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locomotion.rcss_simulation import make_single_t1_simulation  # noqa: E402
from locomotion.t1_spec import (  # noqa: E402
    T1_JOINT_NAMES,
    T1_LOWER_LIMITS,
    T1_MOTOR_NAMES,
    T1_UPPER_LIMITS,
    clamp_t1_targets,
)


def canonical_runtime_name(name: str) -> str:
    """去掉球员实例前缀，并修正模型中头部 yaw 的历史名称。"""

    suffix = name.rsplit("-", 1)[-1]
    return "Head_yaw" if suffix == "AAHead_yaw" else suffix


def main() -> None:
    """逐项检查关节顺序、三类执行器通道、有限控制值和官方限位。"""

    sim, agent = make_single_t1_simulation()

    actual_names = tuple(canonical_runtime_name(joint.name) for joint in agent._joints)
    if actual_names != T1_JOINT_NAMES:
        raise AssertionError(f"关节顺序不匹配\n期望={T1_JOINT_NAMES}\n实际={actual_names}")
    if len(T1_MOTOR_NAMES) != 23:
        raise AssertionError("电机表不是 23 维")

    prefix = agent.agent_id.prefix
    actuator_ids = [mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + motor + "_pos") for motor in T1_MOTOR_NAMES]
    missing = [
        f"{motor}_{channel}"
        for motor in T1_MOTOR_NAMES
        for channel in ("tau", "pos", "vel")
        if mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + motor + f"_{channel}") < 0
    ]
    if missing:
        available = [mujoco.mj_id2name(sim.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(sim.mj_model.nu)]
        raise AssertionError(f"缺少执行器通道：{missing}；前缀={prefix!r}；现有={available}")

    for motor in T1_MOTOR_NAMES:
        sim.ctrl_motor(prefix + motor, 0.0, 0.0, 25.0, 2.0, 0.0)
    if not np.isfinite(sim.mj_data.ctrl).all():
        raise AssertionError("发送 23 路中立目标后出现 NaN/Inf 控制值")

    low = clamp_t1_targets(np.full(23, -999.0, dtype=np.float32))
    high = clamp_t1_targets(np.full(23, 999.0, dtype=np.float32))
    if not (np.array_equal(low, T1_LOWER_LIMITS) and np.array_equal(high, T1_UPPER_LIMITS)):
        raise AssertionError("官方机械限位裁剪不完整")

    print("PASS: T1_23_DOF_CONTRACT")
    for index, (name, motor, actuator_id, lower, upper) in enumerate(
        zip(T1_JOINT_NAMES, T1_MOTOR_NAMES, actuator_ids, T1_LOWER_LIMITS, T1_UPPER_LIMITS, strict=True)
    ):
        print(f"{index:02d} {name:22s} -> {motor:4s} actuator={actuator_id:3d} range_rad=[{lower:+.4f}, {upper:+.4f}]")


if __name__ == "__main__":
    main()
