"""Calibrate collision-clear heights for the 5/10-degree v045 curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v044_mjlab_t1_getup.mjlab_getup_policy import HOME_QPOS  # noqa: E402
from experiments.v045_rcss_t1_getup_adaptation.scripts import (  # noqa: E402
    evaluate_v045_rcss,
)
from locomotion import t1_spec  # noqa: E402
from locomotion.rcss_simulation import make_single_t1_simulation  # noqa: E402


TILT_DEGREES = (5, 10)
ORIENTATION_JITTER_RAD = 0.04
JOINT_JITTER_RAD = 0.035
MINIMUM_CLEARANCE_M = 0.005
ADDITIONAL_RESERVE_M = 0.010


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-pose-level", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=46650)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT
        / "metrics"
        / "v045_fine_tilt_height_calibration_run_01.json",
    )
    args = parser.parse_args()
    if args.samples_per_pose_level < 2000:
        parser.error("at least 2000 samples per pose and fine tilt are required")
    return args


def _sample_required_height(
    simulation: object,
    agent: object,
    *,
    pose: str,
    tilt_deg: int,
    rng: np.random.Generator,
    pitch_id: int,
    robot_geom_ids: list[int],
) -> float:
    axis, signed_full_angle = evaluate_v045_rcss.POSES[pose]
    signed_tilt = math.copysign(math.radians(tilt_deg), signed_full_angle)
    base = evaluate_v045_rcss._axis_angle(axis, signed_tilt)
    yaw = evaluate_v045_rcss._axis_angle(
        (0.0, 0.0, 1.0), float(rng.uniform(-math.pi, math.pi))
    )
    jitter_x = evaluate_v045_rcss._axis_angle(
        (1.0, 0.0, 0.0),
        float(rng.uniform(-ORIENTATION_JITTER_RAD, ORIENTATION_JITTER_RAD)),
    )
    jitter_y = evaluate_v045_rcss._axis_angle(
        (0.0, 1.0, 0.0),
        float(rng.uniform(-ORIENTATION_JITTER_RAD, ORIENTATION_JITTER_RAD)),
    )
    quat = evaluate_v045_rcss._quat_mul(
        yaw,
        evaluate_v045_rcss._quat_mul(
            jitter_y, evaluate_v045_rcss._quat_mul(jitter_x, base)
        ),
    )
    quat /= np.linalg.norm(quat)
    joints = t1_spec.clamp_t1_targets(
        HOME_QPOS
        + rng.uniform(-JOINT_JITTER_RAD, JOINT_JITTER_RAD, size=23).astype(
            np.float32
        )
    ).astype(np.float64)
    agent.place_at((0.0, 0.0, 0.0), tuple(float(value) for value in quat))
    agent.init_joints(pos=joints)
    simulation.mj_data.ctrl[:] = 0.0
    mujoco.mj_forward(simulation.mj_model, simulation.mj_data)
    from_to = np.zeros(6, dtype=np.float64)
    minimum_distance = min(
        float(
            mujoco.mj_geomDistance(
                simulation.mj_model,
                simulation.mj_data,
                pitch_id,
                geom_id,
                10.0,
                from_to,
            )
        )
        for geom_id in robot_geom_ids
    )
    return -minimum_distance + MINIMUM_CLEARANCE_M


def main() -> int:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration evidence: {output}")
    simulation, agent = make_single_t1_simulation()
    simulation.ball.drop_at(20.0, 0.0)
    pitch_id = mujoco.mj_name2id(
        simulation.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "pitch"
    )
    robot_geom_ids = evaluate_v045_rcss._robot_geom_ids(
        simulation.mj_model, agent.agent_id.prefix
    )
    if pitch_id < 0 or not robot_geom_ids:
        raise RuntimeError("RCSS pitch or robot collision geometry is unavailable")

    rng = np.random.default_rng(args.seed)
    per_level: dict[str, object] = {}
    selected: dict[str, list[float]] = {}
    all_clear = True
    for tilt_deg in TILT_DEGREES:
        per_pose: dict[str, object] = {}
        selected_heights: list[float] = []
        for pose in evaluate_v045_rcss.POSES:
            values = np.asarray(
                [
                    _sample_required_height(
                        simulation,
                        agent,
                        pose=pose,
                        tilt_deg=tilt_deg,
                        rng=rng,
                        pitch_id=pitch_id,
                        robot_geom_ids=robot_geom_ids,
                    )
                    for _ in range(args.samples_per_pose_level)
                ],
                dtype=np.float64,
            )
            sampled_max = float(values.max())
            calibrated = math.ceil(
                (sampled_max + ADDITIONAL_RESERVE_M) * 1000.0
            ) / 1000.0
            failures = int(np.sum(values > calibrated))
            all_clear = all_clear and failures == 0
            selected_heights.append(calibrated)
            per_pose[pose] = {
                "samples": int(values.size),
                "minimum_required_root_height_m": float(values.min()),
                "mean_required_root_height_m": float(values.mean()),
                "p99_required_root_height_m": float(np.quantile(values, 0.99)),
                "maximum_required_root_height_m": sampled_max,
                "selected_fixed_root_height_m": calibrated,
                "reserve_over_sampled_max_m": float(calibrated - sampled_max),
                "sampled_clearance_failures_at_selected_height": failures,
            }
        key = str(tilt_deg)
        selected[key] = selected_heights
        per_level[key] = {"tilt_deg": tilt_deg, "per_pose": per_pose}

    script_path = Path(__file__).resolve()
    payload = {
        "schema_version": "v045_fine_tilt_height_calibration_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "rcssservermj_mujoco_3_5_exact_collision_geometry",
        "mujoco_version": mujoco.__version__,
        "seed": int(args.seed),
        "samples_per_pose_level": int(args.samples_per_pose_level),
        "total_samples": int(
            args.samples_per_pose_level
            * len(evaluate_v045_rcss.POSES)
            * len(TILT_DEGREES)
        ),
        "pose_order": list(evaluate_v045_rcss.POSES),
        "tilt_degrees": list(TILT_DEGREES),
        "domain": {
            "yaw_rad": [-math.pi, math.pi],
            "orientation_jitter_rad": [
                -ORIENTATION_JITTER_RAD,
                ORIENTATION_JITTER_RAD,
            ],
            "joint_jitter_rad": [-JOINT_JITTER_RAD, JOINT_JITTER_RAD],
            "minimum_clearance_m": MINIMUM_CLEARANCE_M,
            "additional_sampled_reserve_m": ADDITIONAL_RESERVE_M,
        },
        "selection_rule": "ceil_to_1mm(sampled_max_required_height + 10mm)",
        "per_level": per_level,
        "selected_root_heights_m_by_tilt_deg": selected,
        "all_sampled_resets_clear_at_selected_heights": bool(all_clear),
        "quality_passed": False,
        "scope": (
            "Collision-clear reset proof for non-promotable 5/10-degree "
            "engineering curriculum stages; not recovery-quality evidence."
        ),
        "calibration_script": _reference(script_path),
        "dependencies": [
            _reference(Path(evaluate_v045_rcss.__file__)),
            _reference(Path(t1_spec.__file__)),
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), **payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
