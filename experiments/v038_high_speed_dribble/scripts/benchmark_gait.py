"""逐速检查稳定性、真实速度和双脚腾空比例，区分快走与真正跑步。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v038_high_speed_dribble.curriculum import unique_metrics_path  # noqa: E402
from locomotion.official_t1_policy import MotionCommand  # noqa: E402
from locomotion.rcss_simulation import make_single_t1_simulation, torso_upright  # noqa: E402
from locomotion.rcss_t1_controller import RcssT1Controller  # noqa: E402


def _geom_id_ending_with(model: mujoco.MjModel, suffix: str) -> int:
    matches = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and name.endswith(suffix):
            matches.append(geom_id)
    if len(matches) != 1:
        raise RuntimeError(f"expected one geom ending with {suffix!r}, got {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speeds", nargs="+", type=float, default=[0.45, 0.60, 0.80, 1.00])
    parser.add_argument("--prepare-seconds", type=float, default=2.0)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--ignore-first-seconds", type=float, default=2.0)
    args = parser.parse_args()

    low_level_model = PROJECT_ROOT.parents[1] / "third_party" / "booster_gym" / "deploy" / "models" / "T1.pt"
    simulation, agent = make_single_t1_simulation()
    controller = RcssT1Controller(low_level_model)
    model = simulation.mj_model
    pitch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pitch")
    left_foot = _geom_id_ending_with(model, "left_foot")
    right_foot = _geom_id_ending_with(model, "right_foot")
    dt = float(simulation.timestep)
    rows: list[dict[str, object]] = []

    for requested_speed in args.speeds:
        controller.reset_robot(simulation, agent)
        # 纯步态基准不能让位于中点的足球碰撞机器人并污染速度或接触统计。
        simulation.ball.drop_at(20.0, 0.0)
        mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
        for _ in range(round(args.prepare_seconds / dt)):
            controller.prepare_step(simulation, agent)
            simulation.step([])
        start_x = float(agent.xpos[0])
        velocities: list[float] = []
        flight = double_support = left_only = right_only = 0
        contact_samples = 0
        minimum_height = float(agent.xpos[2])
        minimum_upright = float(torso_upright(agent))
        fell = False
        total_steps = round(args.duration_seconds / dt)
        ignore_steps = round(args.ignore_first_seconds / dt)
        for step in range(total_steps):
            controller.command_step(
                simulation,
                agent,
                MotionCommand(vx=float(requested_speed)),
                step * dt,
            )
            simulation.step([])
            height = float(agent.xpos[2])
            upright = float(torso_upright(agent))
            minimum_height = min(minimum_height, height)
            minimum_upright = min(minimum_upright, upright)
            fell = fell or height < 0.34 or upright < 0.20
            if step < ignore_steps:
                continue

            rotation_world_from_body = np.asarray(agent.xmat, dtype=np.float64).reshape(3, 3)
            body_velocity = rotation_world_from_body.T @ np.asarray(agent._qvel[:3], dtype=np.float64)
            velocities.append(float(body_velocity[0]))
            left_contact = False
            right_contact = False
            for contact_index in range(simulation.mj_data.ncon):
                contact = simulation.mj_data.contact[contact_index]
                pair = {int(contact.geom1), int(contact.geom2)}
                left_contact = left_contact or pair == {pitch, left_foot}
                right_contact = right_contact or pair == {pitch, right_foot}
            contact_samples += 1
            flight += int(not left_contact and not right_contact)
            double_support += int(left_contact and right_contact)
            left_only += int(left_contact and not right_contact)
            right_only += int(right_contact and not left_contact)

        flight_fraction = flight / max(1, contact_samples)
        rows.append(
            {
                "requested_speed_mps": float(requested_speed),
                "mean_body_forward_speed_mps": float(np.mean(velocities)) if velocities else 0.0,
                "distance_m": float(agent.xpos[0] - start_x),
                "flight_fraction": float(flight_fraction),
                "double_support_fraction": double_support / max(1, contact_samples),
                "left_only_fraction": left_only / max(1, contact_samples),
                "right_only_fraction": right_only / max(1, contact_samples),
                "strict_gait_class": "running" if flight_fraction >= 0.02 else "walking_or_fast_walking",
                "fell": bool(fell),
                "minimum_height_m": minimum_height,
                "minimum_upright": minimum_upright,
                "all_23_targets_sent": bool(controller.all_23_targets_sent),
            }
        )

    payload = {
        "version": "v038_high_speed_dribble",
        "definition": "strict running requires at least 2% sampled flight phase with neither foot touching pitch",
        "sim_dt_s": dt,
        "prepare_seconds": float(args.prepare_seconds),
        "duration_seconds": float(args.duration_seconds),
        "results": rows,
    }
    metrics = unique_metrics_path("v038_gait_benchmark")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(metrics)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
