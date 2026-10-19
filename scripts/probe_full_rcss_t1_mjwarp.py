#!/usr/bin/env python3
"""在隔离环境中验证完整 RCSS T1 能否由 MuJoCo Warp 批量步进。

RCSSServerMJ 0.2.1 官方依赖 MuJoCo 3.5，而 MuJoCo Warp 3.11 要求较新
MuJoCo。本脚本专门用于兼容实验；正式比赛环境仍使用项目锁定的 3.5。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import rcsssmj.sim.simulation as simulation_module
import warp as wp


simulation_module.a_forward = lambda *args, **kwargs: None
simulation_module.a_step = lambda *args, **kwargs: None

from rcsssmj.games.soccer.sim.soccer_sim import SoccerSimulation  # noqa: E402
from rcsssmj.games.soccer.soccer_fields import FIFA7vs7SoccerField  # noqa: E402
from rcsssmj.games.soccer.soccer_rules import SSim2026Rules  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AgentParams:
    model_name: str = "T1"
    team_name: str = "QiBlue"
    player_no: int = 1


class RecompiledSoccerSimulation(SoccerSimulation):
    """Recreate MjData after adding T1, matching the project's safe wrapper."""

    def _recompile(self) -> None:
        self._last_recompilation = self._frame_id
        self._mj_model = self._mj_spec.compile()
        self._mj_data = mujoco.MjData(self._mj_model)
        self._mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._a_data = simulation_module.a_recompile(self._mj_spec, self._a_data)
        simulation_module.a_forward(self._a_data, self._mj_data)
        for simulation_object in self.sim_objects:
            simulation_object.bind(self._mj_model, self._mj_data)
        self._update_body_children_map()


def build_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    simulation = RecompiledSoccerSimulation(FIFA7vs7SoccerField(), SSim2026Rules())
    if not simulation.init():
        raise RuntimeError("RCSSServerMJ initialization failed")
    players = simulation.add_players([AgentParams()])
    if len(players) != 1 or players[0] is None:
        raise RuntimeError("failed to add one full T1")
    simulation.mj_data.ctrl[:] = 0.0
    return simulation.mj_model, simulation.mj_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worlds < 1 or args.steps < 1:
        raise ValueError("--worlds and --steps must both be positive")

    wp.init()
    wp.set_device(args.device)
    device = wp.get_device()
    if not device.is_cuda:
        raise RuntimeError(f"GPU probe requires CUDA, got {device}")

    build_started = time.perf_counter()
    host_model, host_data = build_model()
    build_seconds = time.perf_counter() - build_started
    actuator_names = tuple(
        mujoco.mj_id2name(host_model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(host_model.nu)
    )
    triplets_valid = all(
        name.endswith(expected)
        for index, name in enumerate(actuator_names)
        for expected in (("_tau", "_pos", "_vel")[index % 3],)
    )

    compile_started = time.perf_counter()
    warp_model = mjw.put_model(host_model)
    warp_data = mjw.put_data(
        host_model,
        host_data,
        nworld=args.worlds,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
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
    nacon = warp_data.nacon.numpy()
    nefc = warp_data.nefc.numpy()
    finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
    allocation_has_headroom = bool(
        int(nacon.max(initial=0)) < args.nconmax * args.worlds
        and int(nefc.max(initial=0)) < args.njmax
    )
    simulated_seconds = float(args.worlds * args.steps * host_model.opt.timestep)

    result = {
        "kind": "full_rcss_t1_mujoco_warp_compatibility_probe",
        "not_training_evidence": True,
        "dependency_override": "rcsssmj 0.2.1 with mujoco 3.11 instead of pinned 3.5",
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
        "controlled_joint_count": int(host_model.nu // 3),
        "actuator_triplets_valid": triplets_valid,
        "nconmax_per_world": args.nconmax,
        "njmax_per_world": args.njmax,
        "ncon_observed_total": int(nacon.max(initial=0)),
        "nefc_observed_max": int(nefc.max(initial=0)),
        "allocation_has_headroom": allocation_has_headroom,
        "nbody": int(host_model.nbody),
        "ngeom": int(host_model.ngeom),
        "build_seconds": build_seconds,
        "first_compile_and_step_seconds": compile_seconds,
        "measured_step_seconds": step_seconds,
        "aggregate_simulated_seconds": simulated_seconds,
        "aggregate_realtime_factor": (
            simulated_seconds / step_seconds if step_seconds > 0.0 else math.inf
        ),
        "qpos_qvel_finite": finite,
        "qpos_abs_max": float(np.abs(qpos).max(initial=0.0)),
        "qvel_abs_max": float(np.abs(qvel).max(initial=0.0)),
        "compatible": bool(
            finite and host_model.nu == 69 and triplets_valid and allocation_has_headroom
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output.expanduser()
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing probe: {output}")
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["compatible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
