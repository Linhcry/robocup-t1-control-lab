#!/usr/bin/env python3
"""把 RCSSServerMJ 动态组装的完整 23 路 T1 保存为 MuJoCo 二进制模型。

MJB 只保证由相同 MuJoCo 版本读取；它用于留存结构和哈希，不是可跨版本
训练模型，也不是策略检查点。跨版本 GPU 探测使用
``probe_full_rcss_t1_mjwarp.py`` 在目标版本中重新组装模型。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco

from locomotion.rcss_simulation import make_single_t1_simulation
from locomotion.t1_spec import T1_MOTOR_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "records" / "full_rcss_t1_23dof_20260820.mjb",
    )
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output}")

    simulation, _ = make_single_t1_simulation()
    model = simulation.mj_model
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )
    # RCSS exposes three MuJoCo actuators (torque, position and velocity) for
    # every one of the 23 commanded joints, hence model.nu is 69 rather than 23.
    expected_suffixes = tuple(
        suffix
        for motor_name in T1_MOTOR_NAMES
        for suffix in (
            f"-{motor_name}_tau",
            f"-{motor_name}_pos",
            f"-{motor_name}_vel",
        )
    )
    if model.nu != 3 * len(T1_MOTOR_NAMES):
        raise RuntimeError(
            "full RCSS T1 must expose 23 torque/position/velocity triplets, "
            f"got {model.nu} raw actuators"
        )
    if not all(name.endswith(suffix) for name, suffix in zip(actuator_names, expected_suffixes, strict=True)):
        raise RuntimeError("RCSS actuator triplet order differs from locomotion.t1_spec")

    output.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(output), None)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    payload = {
        "kind": "full_rcss_t1_mujoco_binary_export",
        "not_training_evidence": True,
        "model": str(output),
        "sha256": digest,
        "mujoco_version": mujoco.__version__,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "controlled_joint_count": len(T1_MOTOR_NAMES),
        "actuators_per_joint": 3,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "actuator_names": list(actuator_names),
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
