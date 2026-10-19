#!/usr/bin/env python3
"""验证 Booster T1 是否能在官方 MuJoCo Warp 后端批量步进。

这个脚本只做后端兼容性与数值稳定性探测，不训练策略，也不能被当成
起身或足球技能已经达标的证据。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_SOCCER_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML = (
    ROBOT_SOCCER_ROOT
    / "third_party"
    / "booster_gym"
    / "resources"
    / "T1"
    / "T1_locomotion.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--expected-nu",
        type=int,
        help="Expected actuator count; use 12 for Booster locomotion XML and 23 for full RCSS T1",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worlds < 1 or args.steps < 1:
        raise ValueError("--worlds and --steps must both be positive")
    xml_path = args.xml.expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    wp.init()
    wp.set_device(args.device)
    device = wp.get_device()
    if not device.is_cuda:
        raise RuntimeError(f"GPU probe requires a CUDA device, got {device}")

    load_started = time.perf_counter()
    if xml_path.suffix.lower() == ".mjb":
        host_model = mujoco.MjModel.from_binary_path(str(xml_path))
    else:
        host_model = mujoco.MjModel.from_xml_path(str(xml_path))
    host_data = mujoco.MjData(host_model)
    mujoco.mj_resetData(host_model, host_data)
    mujoco.mj_forward(host_model, host_data)
    load_seconds = time.perf_counter() - load_started

    compile_started = time.perf_counter()
    warp_model = mjw.put_model(host_model)
    warp_data = mjw.put_data(host_model, host_data, nworld=args.worlds)
    mjw.step(warp_model, warp_data)
    wp.synchronize()
    compile_seconds = time.perf_counter() - compile_started

    step_started = time.perf_counter()
    for _ in range(args.steps):
        mjw.step(warp_model, warp_data)
    wp.synchronize()
    step_seconds = time.perf_counter() - step_started

    qpos = warp_data.qpos.numpy()
    qvel = warp_data.qvel.numpy()
    finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
    simulated_seconds = float(args.worlds * args.steps * host_model.opt.timestep)
    realtime_factor = simulated_seconds / step_seconds if step_seconds > 0.0 else math.inf

    actuator_count_matches = args.expected_nu is None or host_model.nu == args.expected_nu
    result = {
        "kind": "t1_mujoco_warp_compatibility_probe",
        "not_training_evidence": True,
        "xml": str(xml_path),
        "device": str(device),
        "mujoco_version": mujoco.__version__,
        "mujoco_warp_version": mjw.__version__,
        "warp_version": wp.__version__,
        "worlds": args.worlds,
        "steps_after_warmup": args.steps,
        "timestep_seconds": float(host_model.opt.timestep),
        "nq": int(host_model.nq),
        "nv": int(host_model.nv),
        "nu": int(host_model.nu),
        "expected_nu": args.expected_nu,
        "actuator_count_matches": actuator_count_matches,
        "nbody": int(host_model.nbody),
        "ngeom": int(host_model.ngeom),
        "load_seconds": load_seconds,
        "first_compile_and_step_seconds": compile_seconds,
        "measured_step_seconds": step_seconds,
        "aggregate_simulated_seconds": simulated_seconds,
        "aggregate_realtime_factor": realtime_factor,
        "qpos_qvel_finite": finite,
        "qpos_abs_max": float(np.abs(qpos).max(initial=0.0)),
        "qvel_abs_max": float(np.abs(qvel).max(initial=0.0)),
        "compatible": bool(finite and actuator_count_matches),
    }

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        output_path = args.output.expanduser()
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite existing probe: {output_path}")
        output_path.write_text(output + "\n", encoding="utf-8")
    return 0 if result["compatible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
