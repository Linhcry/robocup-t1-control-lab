"""Audit the official mjlab T1 training model against the RCSS T1 contract.

This is a structural audit only.  Matching joint order, limits and mass does not
prove that a policy trained in one dynamics/actuator model works in the other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.t1_spec import (  # noqa: E402
    T1_JOINT_NAMES,
    T1_LOWER_LIMITS,
    T1_UPPER_LIMITS,
)


DEFAULT_MJLAB_XML = Path(
    "/home/linh/robot_soccer/third_party/mjlab_playground/src/"
    "mjlab_playground/asset_zoo/robots/booster_t1/xmls/t1.xml"
)
_LOCAL_RCSS_MJB = REPO_ROOT / "records" / "full_rcss_t1_23dof_20260820.mjb"
_WINDOWS_MOUNT_RCSS_MJB = Path(
    "/mnt/d/机器人足球/training_runs/rcss_t1_training_20260706/records/"
    "full_rcss_t1_23dof_20260820.mjb"
)
DEFAULT_RCSS_MJB = (
    _LOCAL_RCSS_MJB if _LOCAL_RCSS_MJB.is_file() else _WINDOWS_MOUNT_RCSS_MJB
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "v044_mjlab_t1_getup"
    / "metrics"
    / "v044_t1_mapping_audit_run_02.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_joint_name(name: str) -> str:
    if name.startswith("r-0-1-"):
        name = name[len("r-0-1-") :]
    if name == "AAHead_yaw":
        return "Head_yaw"
    return name


def _hinge_joints(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    limits: list[np.ndarray] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        raw_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        if raw_name is None:
            raise RuntimeError(f"joint {joint_id} has no name")
        names.append(_normalise_joint_name(raw_name))
        limits.append(np.asarray(model.jnt_range[joint_id], dtype=np.float64))
    return names, np.stack(limits, axis=0)


def _rcss_robot_mass(model: mujoco.MjModel) -> float:
    total = 0.0
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is not None and name.startswith("r-0-1-"):
            total += float(model.body_mass[body_id])
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjlab-xml", type=Path, default=DEFAULT_MJLAB_XML)
    parser.add_argument("--rcss-mjb", type=Path, default=DEFAULT_RCSS_MJB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for source in (args.mjlab_xml, args.rcss_mjb):
        if not source.is_file():
            raise FileNotFoundError(source)
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite historical evidence: {args.output}"
        )

    mjlab_model = mujoco.MjModel.from_xml_path(str(args.mjlab_xml))
    rcss_model = mujoco.MjModel.from_binary_path(str(args.rcss_mjb))
    mjlab_names, mjlab_limits = _hinge_joints(mjlab_model)
    rcss_names, rcss_limits = _hinge_joints(rcss_model)

    canonical_limits = np.stack(
        [T1_LOWER_LIMITS, T1_UPPER_LIMITS], axis=-1
    ).astype(np.float64)
    mjlab_mass = float(np.sum(mjlab_model.body_mass))
    rcss_mass = _rcss_robot_mass(rcss_model)
    rcss_actuator_names = [
        mujoco.mj_id2name(rcss_model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(rcss_model.nu)
    ]
    actuator_triplets_ok = rcss_model.nu == 69 and all(
        rcss_actuator_names[index * 3 + offset]
        == f"r-0-1-{motor}_{suffix}"
        for index, motor in enumerate(
            (
                "he1", "he2", "lae1", "lae2", "lae3", "lae4",
                "rae1", "rae2", "rae3", "rae4", "te1", "lle1",
                "lle2", "lle3", "lle4", "lle5", "lle6", "rle1",
                "rle2", "rle3", "rle4", "rle5", "rle6",
            )
        )
        for offset, suffix in enumerate(("tau", "pos", "vel"))
    )

    checks = {
        "mjlab_joint_count_23": len(mjlab_names) == 23,
        "rcss_joint_count_23": len(rcss_names) == 23,
        "mjlab_order_matches_canonical": tuple(mjlab_names) == T1_JOINT_NAMES,
        "rcss_order_matches_canonical": tuple(rcss_names) == T1_JOINT_NAMES,
        "mjlab_and_rcss_limits_exact": bool(
            np.array_equal(mjlab_limits, rcss_limits)
        ),
        "mjlab_limits_within_0_007_rad_of_canonical": bool(
            np.max(np.abs(mjlab_limits - canonical_limits)) <= 0.007
        ),
        "robot_mass_matches_within_1e_6_kg": abs(mjlab_mass - rcss_mass) <= 1e-6,
        "rcss_has_23_tau_pos_vel_actuator_triplets": actuator_triplets_ok,
    }
    mapping_passed = all(checks.values())
    result = {
        "schema_version": "v044_t1_mapping_audit_v1",
        "sources": {
            "mjlab_xml": {
                "path": str(args.mjlab_xml),
                "sha256": _sha256(args.mjlab_xml),
                "mujoco_timestep_s": float(mjlab_model.opt.timestep),
            },
            "rcss_mjb": {
                "path": str(args.rcss_mjb),
                "sha256": _sha256(args.rcss_mjb),
                "mujoco_timestep_s": float(rcss_model.opt.timestep),
            },
        },
        "joint_order": {
            "canonical": list(T1_JOINT_NAMES),
            "mjlab": mjlab_names,
            "rcss": rcss_names,
        },
        "maximum_limit_delta_rad": {
            "mjlab_vs_rcss": float(np.max(np.abs(mjlab_limits - rcss_limits))),
            "mjlab_vs_canonical": float(
                np.max(np.abs(mjlab_limits - canonical_limits))
            ),
        },
        "robot_mass_kg": {
            "mjlab": mjlab_mass,
            "rcss": rcss_mass,
            "absolute_delta": abs(mjlab_mass - rcss_mass),
        },
        "actuator_semantics": {
            "mjlab": "23 builtin position actuators added by mjlab configuration",
            "rcss": "69 actuators: torque, position and velocity per joint",
            "dynamics_equivalent": False,
        },
        "checks": checks,
        "mapping_passed": mapping_passed,
        "deployment_ready": False,
        "quality_passed": False,
        "conclusion": (
            "Structural mapping passed; RCSS closed-loop recovery evaluation is still required."
            if mapping_passed
            else "Structural mapping failed; checkpoint export and deployment are blocked."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "mapping_passed": mapping_passed,
        "deployment_ready": False,
    }, ensure_ascii=False))
    return 0 if mapping_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
