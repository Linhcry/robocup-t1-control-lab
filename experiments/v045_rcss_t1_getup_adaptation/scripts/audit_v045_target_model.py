#!/usr/bin/env python3
"""Fail-closed semantic audit for the exact-RCSS v045 training robot.

The exported RCSS MJB was produced by MuJoCo 3.5 while the isolated MJLab
environment uses MuJoCo 3.11.  MJB files are version-specific, so the main
process asks the RCSS Python environment to read the MJB and return a portable
JSON snapshot.  The comparison itself then runs in the MJLab environment.

This is structural evidence only.  Passing it does not prove recovery quality
or authorize promotion of a policy checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RCSS_PYTHON = Path(
    "/home/linh/robot_soccer/envs/rl-mujoco/bin/python"
)
_LOCAL_MJB = REPO_ROOT / "records" / "full_rcss_t1_23dof_20260820.mjb"
_WINDOWS_MOUNT_MJB = Path(
    "/mnt/d/机器人足球/training_runs/rcss_t1_training_20260706/records/"
    "full_rcss_t1_23dof_20260820.mjb"
)
DEFAULT_RCSS_MJB = _LOCAL_MJB if _LOCAL_MJB.is_file() else _WINDOWS_MOUNT_MJB
PREFIX = "r-0-1-"
JERSEY_GEOMS = ("jersey", "jersey-front")
ABSOLUTE_TOLERANCE = 1e-9
DEFAULT_SOURCE_ACTION_AUDIT = (
    EXPERIMENT_ROOT / "metrics" / "v045_source_action_semantics_audit_run_01.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_name(name: str | None, prefix: str | None) -> str | None:
    if name is None:
        return None
    if prefix and name.startswith(prefix):
        name = name[len(prefix) :]
    if name == "AAHead_yaw":
        return "Head_yaw"
    return name


def _model_snapshot(model: Any, mujoco: Any, *, prefix: str | None) -> dict[str, Any]:
    """Create a JSON-safe snapshot of one robot, excluding the world."""

    body_ids: list[int] = []
    bodies: list[dict[str, Any]] = []
    for body_id in range(1, model.nbody):
        raw_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        if prefix is not None and not (raw_name or "").startswith(prefix):
            continue
        body_ids.append(body_id)
        bodies.append(
            {
                "name": _normalise_name(raw_name, prefix),
                "mass": float(model.body_mass[body_id]),
                "inertia": [float(value) for value in model.body_inertia[body_id]],
            }
        )
    body_id_set = set(body_ids)

    joints: list[dict[str, Any]] = []
    for joint_id in range(model.njnt):
        raw_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        if prefix is not None and not (raw_name or "").startswith(prefix):
            continue
        joints.append(
            {
                "name": _normalise_name(raw_name, prefix),
                "type": int(model.jnt_type[joint_id]),
                "range": [float(value) for value in model.jnt_range[joint_id]],
                "damping": float(model.dof_damping[int(model.jnt_dofadr[joint_id])]),
                "armature": float(model.dof_armature[int(model.jnt_dofadr[joint_id])]),
                "frictionloss": float(
                    model.dof_frictionloss[int(model.jnt_dofadr[joint_id])]
                ),
                "actuator_force_range": [
                    float(value) for value in model.jnt_actfrcrange[joint_id]
                ],
            }
        )

    geoms: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id not in body_id_set:
            continue
        raw_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        geoms.append(
            {
                "name": _normalise_name(raw_name, prefix),
                "body": _normalise_name(body_name, prefix),
                "type": int(model.geom_type[geom_id]),
                "size": [float(value) for value in model.geom_size[geom_id]],
                "pos": [float(value) for value in model.geom_pos[geom_id]],
                "quat": [float(value) for value in model.geom_quat[geom_id]],
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "condim": int(model.geom_condim[geom_id]),
                "friction": [float(value) for value in model.geom_friction[geom_id]],
                "solref": [float(value) for value in model.geom_solref[geom_id]],
                "solimp": [float(value) for value in model.geom_solimp[geom_id]],
            }
        )

    actuators: list[dict[str, Any]] = []
    for actuator_id in range(model.nu):
        raw_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        if prefix is not None and not (raw_name or "").startswith(prefix):
            continue
        target_joint_id = int(model.actuator_trnid[actuator_id, 0])
        target_joint_name = None
        if 0 <= target_joint_id < model.njnt:
            target_joint_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, target_joint_id
            )
        actuators.append(
            {
                "name": _normalise_name(raw_name, prefix),
                "target_joint": _normalise_name(target_joint_name, prefix),
                "transmission_type": int(model.actuator_trntype[actuator_id]),
                "dynamics_type": int(model.actuator_dyntype[actuator_id]),
                "gain_type": int(model.actuator_gaintype[actuator_id]),
                "bias_type": int(model.actuator_biastype[actuator_id]),
                "control_limited": bool(model.actuator_ctrllimited[actuator_id]),
                "force_limited": bool(model.actuator_forcelimited[actuator_id]),
                "control_range": [
                    float(value) for value in model.actuator_ctrlrange[actuator_id]
                ],
                "force_range": [
                    float(value) for value in model.actuator_forcerange[actuator_id]
                ],
                "gear": [float(value) for value in model.actuator_gear[actuator_id]],
                "gain_parameters": [
                    float(value) for value in model.actuator_gainprm[actuator_id]
                ],
                "bias_parameters": [
                    float(value) for value in model.actuator_biasprm[actuator_id]
                ],
            }
        )

    return {
        "mujoco_timestep_s": float(model.opt.timestep),
        "solver_options": {
            "timestep": float(model.opt.timestep),
            "integrator": int(model.opt.integrator),
            "impratio": float(model.opt.impratio),
            "cone": int(model.opt.cone),
            "jacobian": int(model.opt.jacobian),
            "solver": int(model.opt.solver),
            "iterations": int(model.opt.iterations),
            "tolerance": float(model.opt.tolerance),
            "ls_iterations": int(model.opt.ls_iterations),
            "ls_tolerance": float(model.opt.ls_tolerance),
            "ccd_iterations": int(model.opt.ccd_iterations),
        },
        "robot_body_count": len(bodies),
        "robot_geom_count": len(geoms),
        "robot_joint_count_including_free": len(joints),
        "robot_actuator_count": len(actuators),
        "robot_mass_kg": sum(body["mass"] for body in bodies),
        "bodies": bodies,
        "joints": joints,
        "geoms": geoms,
        "actuators": actuators,
    }


def _snapshot_mjb(path: Path) -> int:
    """Read the version-specific MJB inside the RCSS MuJoCo environment."""

    import mujoco

    if not path.is_file():
        raise FileNotFoundError(path)
    model = mujoco.MjModel.from_binary_path(str(path))
    payload = {
        "mujoco_version": mujoco.__version__,
        "snapshot": _model_snapshot(model, mujoco, prefix=PREFIX),
    }
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


def _read_mjb_snapshot(mjb: Path, rcss_python: Path) -> dict[str, Any]:
    if not mjb.is_file():
        raise FileNotFoundError(mjb)
    if not rcss_python.is_file():
        raise FileNotFoundError(rcss_python)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        [
            str(rcss_python),
            str(Path(__file__).resolve()),
            "--snapshot-mjb",
            str(mjb.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "RCSS MJB snapshot failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"RCSS MJB snapshot did not return JSON: {completed.stdout!r}"
        ) from error


def _close(left: Any, right: Any, tolerance: float = ABSOLUTE_TOLERANCE) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _records_match(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        all(_close(a[key], b[key]) for key in keys)
        for a, b in zip(left, right, strict=True)
    )


def _hinge_joints(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    # MuJoCo enum values: free=0, ball=1, slide=2, hinge=3.
    return [joint for joint in snapshot["joints"] if joint["type"] == 3]


def _raw_triplets_valid(
    actuators: list[dict[str, Any]],
    joint_names: tuple[str, ...],
    motor_names: tuple[str, ...],
) -> bool:
    if len(actuators) != 3 * len(joint_names):
        return False
    expected_bias = {
        "tau": (0.0, 0.0, 0.0),
        "pos": (0.0, -1.0, 0.0),
        "vel": (0.0, 0.0, -1.0),
    }
    for joint_index, (joint_name, motor_name) in enumerate(
        zip(joint_names, motor_names, strict=True)
    ):
        for offset, suffix in enumerate(("tau", "pos", "vel")):
            actuator = actuators[3 * joint_index + offset]
            if actuator["name"] != f"{motor_name}_{suffix}":
                return False
            if actuator["target_joint"] != joint_name:
                return False
            if actuator["transmission_type"] != 0:
                return False
            if actuator["dynamics_type"] != 0 or actuator["gain_type"] != 0:
                return False
            if not _close(actuator["gear"][:6], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
                return False
            if not _close(actuator["gain_parameters"][0], 1.0):
                return False
            if not _close(actuator["bias_parameters"][:3], list(expected_bias[suffix])):
                return False
    return True


def _v045_servos_valid(
    actuators: list[dict[str, Any]],
    joint_names: tuple[str, ...],
    expected_kp: list[float],
    expected_kd: list[float],
    expected_effort: list[float],
) -> bool:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for actuator in actuators:
        by_target.setdefault(actuator["target_joint"], []).append(actuator)
    if set(by_target) != set(joint_names):
        return False
    if not (
        len(expected_kp)
        == len(expected_kd)
        == len(expected_effort)
        == len(joint_names)
    ):
        return False
    expected = {
        name: values
        for name, values in zip(
            joint_names,
            zip(expected_kp, expected_kd, expected_effort, strict=True),
            strict=True,
        )
    }
    for joint_name in joint_names:
        matches = by_target[joint_name]
        if len(matches) != 1:
            return False
        actuator = matches[0]
        kp, kd, effort = expected[joint_name]
        if actuator["transmission_type"] != 0:
            return False
        if actuator["dynamics_type"] != 0 or actuator["gain_type"] != 0:
            return False
        if actuator["bias_type"] != 1:
            return False
        if actuator["control_limited"] or not actuator["force_limited"]:
            return False
        if not _close(actuator["gear"][:6], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
            return False
        if not _close(actuator["gain_parameters"][0], kp, tolerance=1.0e-6):
            return False
        if not _close(
            actuator["bias_parameters"][:3],
            [0.0, -kp, -kd],
            tolerance=1.0e-6,
        ):
            return False
        if not _close(actuator["force_range"], [-effort, effort]):
            return False
    return True


def _next_output() -> Path:
    metrics = EXPERIMENT_ROOT / "metrics"
    for run in range(1, 10_000):
        candidate = metrics / f"v045_target_model_audit_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v045 target-model audit filename remains")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rcss-mjb", type=Path, default=DEFAULT_RCSS_MJB)
    parser.add_argument(
        "--rcss-python",
        type=Path,
        default=Path(os.environ.get("RCSS_AUDIT_PYTHON", DEFAULT_RCSS_PYTHON)),
    )
    parser.add_argument(
        "--source-action-audit", type=Path, default=DEFAULT_SOURCE_ACTION_AUDIT
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.resolve() if args.output else _next_output()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical evidence: {output}")
    source_action_audit_path = args.source_action_audit.resolve()
    if not source_action_audit_path.is_file():
        raise FileNotFoundError(source_action_audit_path)
    source_action_audit = json.loads(
        source_action_audit_path.read_text(encoding="utf-8")
    )
    if (
        source_action_audit.get("schema_version")
        != "v045_source_action_semantics_audit_v1"
        or not source_action_audit.get("action_semantics_confirmed", False)
        or source_action_audit.get("official_runner_clip_actions") is not None
        or source_action_audit.get("quality_passed") is not False
    ):
        raise ValueError("source-action audit is incomplete or invalid")
    action_audit_script = source_action_audit.get("audit_script", {})
    action_audit_script_path = Path(action_audit_script.get("path", "")).resolve()
    if (
        not action_audit_script_path.is_file()
        or _sha256(action_audit_script_path) != action_audit_script.get("sha256")
    ):
        raise ValueError("source-action audit script file/hash mismatch")

    import mujoco

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from experiments.v045_rcss_t1_getup_adaptation.rcss_mjlab_task import (
        RCSS_T1_XML,
        booster_t1_getup_ppo_runner_cfg,
        get_rcss_t1_robot_cfg,
        rcss_t1_getup_env_cfg,
        rcss_t1_getup_ppo_runner_cfg,
    )
    from experiments.v045_rcss_t1_getup_adaptation import reset_mdp
    from experiments.v045_rcss_t1_getup_adaptation import safe_actions
    from experiments.v045_rcss_t1_getup_adaptation import stability_watchdog
    from experiments.v045_rcss_t1_getup_adaptation import training_profiles
    from experiments.v045_rcss_t1_getup_adaptation.actuator_contract import (
        RCSS_EFFECTIVE_EFFORT_LIMIT,
        V045_GETUP_KD,
        V045_GETUP_KP,
    )
    from experiments.v044_mjlab_t1_getup.mjlab_getup_policy import (
        MjlabGetupPolicy,
    )
    from locomotion.t1_spec import (
        T1_ACTION_DELTA_LIMIT_RAD,
        T1_JOINT_NAMES,
        T1_LOWER_LIMITS,
        T1_MOTOR_NAMES,
        T1_TARGET_LIMIT_MARGIN_RAD,
        T1_UPPER_LIMITS,
        T1_VELOCITY_BRAKE_LOOKAHEAD_S,
    )

    if not RCSS_T1_XML.is_file():
        raise FileNotFoundError(RCSS_T1_XML)
    source_model = mujoco.MjModel.from_xml_path(str(RCSS_T1_XML))
    entity = get_rcss_t1_robot_cfg().build()
    v045_model = entity.compile()
    source = _model_snapshot(source_model, mujoco, prefix=None)
    v045 = _model_snapshot(v045_model, mujoco, prefix=None)
    mjb_payload = _read_mjb_snapshot(args.rcss_mjb, args.rcss_python)
    mjb = mjb_payload["snapshot"]

    canonical_names = tuple(T1_JOINT_NAMES)
    source_hinges = _hinge_joints(source)
    v045_hinges = _hinge_joints(v045)
    mjb_hinges = _hinge_joints(mjb)
    source_hinge_names = tuple(joint["name"] for joint in source_hinges)
    v045_hinge_names = tuple(joint["name"] for joint in v045_hinges)
    mjb_hinge_names = tuple(joint["name"] for joint in mjb_hinges)
    canonical_limits = [
        [float(lower), float(upper)]
        for lower, upper in zip(T1_LOWER_LIMITS, T1_UPPER_LIMITS, strict=True)
    ]
    source_limits = [joint["range"] for joint in source_hinges]
    v045_limits = [joint["range"] for joint in v045_hinges]
    mjb_limits = [joint["range"] for joint in mjb_hinges]

    passive_keys = (
        "name",
        "range",
        "damping",
        "armature",
        "frictionloss",
        "actuator_force_range",
    )
    body_keys = ("name", "mass", "inertia")
    geom_keys = (
        "name",
        "body",
        "type",
        "size",
        "pos",
        "quat",
        "contype",
        "conaffinity",
        "condim",
        "friction",
        "solref",
        "solimp",
    )
    mjb_jerseys = [geom for geom in mjb["geoms"] if geom["name"] in JERSEY_GEOMS]
    mjb_base_geoms = [
        geom for geom in mjb["geoms"] if geom["name"] not in JERSEY_GEOMS
    ]
    source_caps = [joint["actuator_force_range"] for joint in source_hinges]
    v045_caps = [joint["actuator_force_range"] for joint in v045_hinges]
    mjb_caps = [joint["actuator_force_range"] for joint in mjb_hinges]
    v045_actuator_caps_by_joint = {
        actuator["target_joint"]: max(abs(value) for value in actuator["force_range"])
        for actuator in v045["actuators"]
    }
    v045_actuators_by_joint = {
        actuator["target_joint"]: actuator for actuator in v045["actuators"]
    }
    v045_compiled_kp = [
        float(v045_actuators_by_joint[name]["gain_parameters"][0])
        for name in canonical_names
    ]
    v045_compiled_kd = [
        float(-v045_actuators_by_joint[name]["bias_parameters"][2])
        for name in canonical_names
    ]
    v045_effective_caps = [
        [
            -min(
                max(abs(value) for value in joint_cap),
                v045_actuator_caps_by_joint[name],
            ),
            min(
                max(abs(value) for value in joint_cap),
                v045_actuator_caps_by_joint[name],
            ),
        ]
        for name, joint_cap in zip(canonical_names, v045_caps, strict=True)
    ]
    deployment_caps = [
        [-float(limit), float(limit)] for limit in RCSS_EFFECTIVE_EFFORT_LIMIT
    ]
    maximum_canonical_limit_delta = max(
        abs(actual - expected)
        for actual_pair, expected_pair in zip(
            v045_limits, canonical_limits, strict=True
        )
        for actual, expected in zip(actual_pair, expected_pair, strict=True)
    )

    action_ids, raw_action_names = entity.find_joints_by_actuator_names((".*",))
    action_names = tuple(
        _normalise_name(name, None) for name in raw_action_names
    )
    training_cfg = rcss_t1_getup_env_cfg()
    training_profile = training_profiles.apply_training_profiles(training_cfg)
    runner_cfg = rcss_t1_getup_ppo_runner_cfg()
    official_source_runner_cfg = booster_t1_getup_ppo_runner_cfg()
    training_sensor_names = tuple(
        sensor.name for sensor in (training_cfg.scene.sensors or ())
    )
    reset_cfg = training_cfg.events["reset_fallen_or_standing"]
    reset_params = reset_cfg.params
    expected_mjb_solver = {
        "timestep": 0.005,
        "integrator": int(mujoco.mjtIntegrator.mjINT_EULER),
        "impratio": 1.0,
        "cone": int(mujoco.mjtCone.mjCONE_PYRAMIDAL),
        "jacobian": int(mujoco.mjtJacobian.mjJAC_AUTO),
        "solver": int(mujoco.mjtSolver.mjSOL_NEWTON),
        "iterations": 100,
        "tolerance": 1.0e-8,
        "ls_iterations": 50,
        "ls_tolerance": 0.01,
        "ccd_iterations": 35,
    }
    checks = {
        "source_xml_has_23_hinge_joints": len(source_hinges) == 23,
        "v045_has_23_hinge_joints": len(v045_hinges) == 23,
        "rcss_mjb_has_23_hinge_joints": len(mjb_hinges) == 23,
        "source_xml_joint_order_matches_canonical": source_hinge_names == canonical_names,
        "v045_joint_order_matches_canonical": v045_hinge_names == canonical_names,
        "rcss_mjb_joint_order_matches_canonical": mjb_hinge_names == canonical_names,
        "source_v045_mjb_joint_limits_match": (
            _close(source_limits, v045_limits) and _close(source_limits, mjb_limits)
        ),
        "joint_limits_within_0_007_rad_of_canonical_contract": (
            maximum_canonical_limit_delta <= 0.007
        ),
        "source_v045_mjb_joint_passive_dynamics_match": (
            _records_match(source_hinges, v045_hinges, keys=passive_keys)
            and _records_match(source_hinges, mjb_hinges, keys=passive_keys)
        ),
        "source_v045_mjb_robot_body_count_is_24": (
            source["robot_body_count"] == 24
            and v045["robot_body_count"] == 24
            and mjb["robot_body_count"] == 24
        ),
        "source_v045_mjb_body_mass_and_inertia_match": (
            _records_match(source["bodies"], v045["bodies"], keys=body_keys)
            and _records_match(source["bodies"], mjb["bodies"], keys=body_keys)
        ),
        "source_and_v045_geom_count_is_36": (
            source["robot_geom_count"] == 36 and v045["robot_geom_count"] == 36
        ),
        "rcss_mjb_has_36_base_geoms_plus_two_jersey_geoms": (
            mjb["robot_geom_count"] == 38
            and len(mjb_base_geoms) == 36
            and tuple(geom["name"] for geom in mjb_jerseys) == JERSEY_GEOMS
        ),
        "source_v045_mjb_base_geom_semantics_match": (
            _records_match(source["geoms"], v045["geoms"], keys=geom_keys)
            and _records_match(source["geoms"], mjb_base_geoms, keys=geom_keys)
        ),
        "source_xml_has_canonical_tau_pos_vel_triplets": _raw_triplets_valid(
            source["actuators"], canonical_names, tuple(T1_MOTOR_NAMES)
        ),
        "rcss_mjb_has_canonical_tau_pos_vel_triplets": _raw_triplets_valid(
            mjb["actuators"], canonical_names, tuple(T1_MOTOR_NAMES)
        ),
        "v045_has_one_position_servo_per_canonical_joint": _v045_servos_valid(
            v045["actuators"],
            canonical_names,
            V045_GETUP_KP.tolist(),
            V045_GETUP_KD.tolist(),
            RCSS_EFFECTIVE_EFFORT_LIMIT.tolist(),
        ),
        "v045_policy_action_order_matches_canonical": (
            tuple(action_ids) == tuple(range(23))
            and action_names == canonical_names
        ),
        "v045_effective_joint_torque_caps_match_rcss": (
            _close(source_caps, v045_caps) and _close(source_caps, mjb_caps)
        ),
        "v045_position_servos_cover_rcss_mechanical_caps": all(
            v045_actuator_caps_by_joint[name]
            >= max(abs(value) for value in joint_cap) - ABSOLUTE_TOLERANCE
            for name, joint_cap in zip(canonical_names, v045_caps, strict=True)
        ),
        "v045_training_and_deployment_effective_caps_match": (
            _close(v045_effective_caps, deployment_caps)
            and _close(mjb_caps, deployment_caps)
        ),
        "v045_training_and_deployment_getup_gains_match": (
            _close(
                v045_compiled_kp, V045_GETUP_KP.tolist(), tolerance=1.0e-6
            )
            and _close(
                v045_compiled_kd, V045_GETUP_KD.tolist(), tolerance=1.0e-6
            )
        ),
        "v045_training_observes_head_hand_and_torso_ground_impacts": all(
            name in training_sensor_names
            for name in (
                "head_ground_contact",
                "hand_ground_contact",
                "torso_ground_contact",
                "left_foot_ground_contact",
                "right_foot_ground_contact",
            )
        ),
        "v045_training_penalizes_contract_impacts_and_torque_saturation": (
            "unsafe_ground_impact" in training_cfg.rewards
            and training_cfg.rewards["unsafe_ground_impact"].weight == -2.0
            and training_cfg.rewards["unsafe_ground_impact"].params
            == {"force_limit_n": 900.0}
            and "torque_saturation" in training_cfg.rewards
            and training_cfg.rewards["torque_saturation"].weight == -0.05
            and training_cfg.rewards["torque_saturation"].params
            == {"fraction": 0.98}
        ),
        "v045_training_rewards_formal_stable_standing": (
            "formal_stable_stand" in training_cfg.rewards
            and training_cfg.rewards["formal_stable_stand"].weight == 5.0
            and training_cfg.rewards["formal_stable_stand"].params
            == {
                "desired_height_m": 0.60,
                "upright_score_min": 0.90,
                "root_speed_mps_max": 0.45,
            }
            and training_cfg.rewards["posture"].params[
                "orientation_threshold"
            ]
            == 0.05
        ),
        "v045_reset_uses_exact_four_pose_family": (
            reset_cfg.func is reset_mdp.reset_rcss_four_pose
            and reset_params
            == {
                "fall_probability": 1.0,
                "orientation_jitter_rad": 0.04,
                "joint_jitter_rad": 0.035,
            }
            and tuple(reset_mdp._ROOT_HEIGHTS_M)
            == (0.237, 0.135, 0.295, 0.297)
        ),
        "v045_reset_heights_are_bound_to_exact_rcss_calibration": (
            training_profile["actual"]["reset_height_calibration"][
                "schema_version"
            ]
            == training_profiles.RESET_CALIBRATION_SCHEMA
            and training_profile["actual"]["reset_height_calibration"][
                "samples_per_pose"
            ]
            == 5000
            and training_profile["actual"]["reset_height_calibration"][
                "all_sampled_resets_clear_at_selected_heights"
            ]
            is True
        ),
        "v045_training_uses_exact_rcss_solver_profile": (
            training_profile["selected"]["solver"] == "rcss_mjb"
            and training_profile["actual"]["solver"]
            == training_profiles.SOLVER_PROFILES["rcss_mjb"]
            and _close(mjb["solver_options"], expected_mjb_solver)
            and training_profile["actual"]["policy_dt_s"] == 0.02
        ),
        "v045_default_target_com_is_not_randomized": (
            training_profile["selected"]["base_com"] == "off"
            and training_profile["actual"]["base_com"] is None
            and "base_com" not in training_cfg.events
        ),
        "v045_watchdog_is_true_substep_and_fail_closed": (
            stability_watchdog.StabilityWatchdogCfg().ring_size_substeps >= 100
            and stability_watchdog.StabilityWatchdogCfg()
            .max_compact_rows_per_event
            >= stability_watchdog.StabilityWatchdogCfg()
            .max_full_snapshots_per_event
            and stability_watchdog.WATCHDOG_SCHEMA_VERSION
            == "v045_physics_substep_watchdog_v2"
            and stability_watchdog.REPORT_SCHEMA_VERSION
            == "v045_stability_report_v2"
            and hasattr(
                safe_actions.RcssSafeRelativeJointPositionAction, "raw_action"
            )
        ),
        "v045_reset_has_immediate_policy_takeover": (
            isinstance(
                training_cfg.actions["joint_pos"],
                safe_actions.RcssSafeRelativeJointPositionActionCfg,
            )
            and training_cfg.terminations["energy"].params["settle_steps"] == 0
        ),
        "v045_raw_action_semantics_match_official_source_and_deployment": (
            official_source_runner_cfg.clip_actions is None
            and runner_cfg.clip_actions is None
            and MjlabGetupPolicy.raw_action_clip is None
            and training_cfg.actions["joint_pos"].scale == 0.6
            and training_cfg.actions["joint_pos"].clip is None
            and source_action_audit["observed_raw_actions"][
                "fraction_above_absolute_one"
            ]
            > 0.0
        ),
        "v045_training_and_deployment_share_dynamic_target_safety": (
            safe_actions.ACTION_DELTA_LIMIT_RAD == T1_ACTION_DELTA_LIMIT_RAD
            == 0.06
            and safe_actions.TARGET_LIMIT_MARGIN_RAD
            == T1_TARGET_LIMIT_MARGIN_RAD
            == 0.08
            and safe_actions.VELOCITY_BRAKE_LOOKAHEAD_S
            == T1_VELOCITY_BRAKE_LOOKAHEAD_S
            == 0.025
        ),
        "v045_observations_fail_on_exact_nonfinite_term": all(
            training_cfg.observations[group].nan_policy == "error"
            and training_cfg.observations[group].nan_check_per_term
            for group in ("actor", "critic")
        ),
    }
    semantics_passed = bool(checks) and all(checks.values())
    result = {
        "schema_version": "v045_target_model_semantic_audit_v10",
        "audit_completed": True,
        "sources": {
            "v045_task": {
                "path": str(EXPERIMENT_ROOT / "rcss_mjlab_task.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "rcss_mjlab_task.py"),
            },
            "v045_training_safety_terms": {
                "path": str(EXPERIMENT_ROOT / "safety_mdp.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "safety_mdp.py"),
            },
            "v045_training_reset_terms": {
                "path": str(EXPERIMENT_ROOT / "reset_mdp.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "reset_mdp.py"),
            },
            "v045_training_safe_action": {
                "path": str(EXPERIMENT_ROOT / "safe_actions.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "safe_actions.py"),
            },
            "v045_training_profiles": {
                "path": str(EXPERIMENT_ROOT / "training_profiles.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "training_profiles.py"),
            },
            "v045_physics_substep_watchdog": {
                "path": str(EXPERIMENT_ROOT / "stability_watchdog.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "stability_watchdog.py"),
            },
            "v045_reset_height_calibration": {
                "path": training_profile["actual"]["reset_height_calibration"][
                    "path"
                ],
                "sha256": training_profile["actual"]["reset_height_calibration"][
                    "sha256"
                ],
                "schema_version": training_profiles.RESET_CALIBRATION_SCHEMA,
            },
            "v045_passive_soft_joint_constraint_probe": {
                "path": training_profile["actual"][
                    "passive_soft_joint_constraint_probe"
                ]["path"],
                "sha256": training_profile["actual"][
                    "passive_soft_joint_constraint_probe"
                ]["sha256"],
                "schema_version": training_profiles.PASSIVE_LIMIT_PROBE_SCHEMA,
            },
            "source_action_semantics_audit": {
                "path": str(source_action_audit_path),
                "sha256": _sha256(source_action_audit_path),
            },
            "rcss_getup_deployment_controller": {
                "path": str(
                    REPO_ROOT
                    / "experiments"
                    / "v045_rcss_t1_getup_adaptation"
                    / "rcss_getup_controller.py"
                ),
                "sha256": _sha256(
                    REPO_ROOT
                    / "experiments"
                    / "v045_rcss_t1_getup_adaptation"
                    / "rcss_getup_controller.py"
                ),
            },
            "v045_actuator_contract": {
                "path": str(EXPERIMENT_ROOT / "actuator_contract.py"),
                "sha256": _sha256(EXPERIMENT_ROOT / "actuator_contract.py"),
            },
            "rcss_getup_deployment_policy": {
                "path": str(
                    REPO_ROOT
                    / "experiments"
                    / "v044_mjlab_t1_getup"
                    / "mjlab_getup_policy.py"
                ),
                "sha256": _sha256(
                    REPO_ROOT
                    / "experiments"
                    / "v044_mjlab_t1_getup"
                    / "mjlab_getup_policy.py"
                ),
            },
            "canonical_t1_control_contract": {
                "path": str(REPO_ROOT / "locomotion" / "t1_spec.py"),
                "sha256": _sha256(REPO_ROOT / "locomotion" / "t1_spec.py"),
            },
            "rcss_robot_xml": {
                "path": str(RCSS_T1_XML),
                "sha256": _sha256(RCSS_T1_XML),
                "mujoco_version": mujoco.__version__,
            },
            "rcss_match_mjb": {
                "path": str(args.rcss_mjb.resolve()),
                "sha256": _sha256(args.rcss_mjb),
                "mujoco_version": mjb_payload["mujoco_version"],
                "reader_python": str(args.rcss_python),
            },
        },
        "counts": {
            "source_xml": {
                "bodies": source["robot_body_count"],
                "geoms": source["robot_geom_count"],
                "hinge_joints": len(source_hinges),
                "actuators": source["robot_actuator_count"],
            },
            "v045_compiled_entity": {
                "bodies": v045["robot_body_count"],
                "geoms": v045["robot_geom_count"],
                "hinge_joints": len(v045_hinges),
                "actuators": v045["robot_actuator_count"],
            },
            "rcss_match_robot": {
                "bodies": mjb["robot_body_count"],
                "geoms_including_jersey": mjb["robot_geom_count"],
                "base_geoms": len(mjb_base_geoms),
                "jersey_geoms": len(mjb_jerseys),
                "hinge_joints": len(mjb_hinges),
                "actuators": mjb["robot_actuator_count"],
            },
        },
        "joint_order": {
            "canonical": list(canonical_names),
            "source_xml": list(source_hinge_names),
            "v045": list(v045_hinge_names),
            "rcss_mjb": list(mjb_hinge_names),
            "v045_policy_action_targets": list(action_names),
            "v045_compiled_actuator_internal_order": [
                actuator["target_joint"] for actuator in v045["actuators"]
            ],
            "internal_order_note": (
                "MJLab stores grouped actuator objects internally, while the action "
                "manager intentionally exposes the natural 23-joint order."
            ),
        },
        "maximum_limit_delta_rad": {
            "source_xml_vs_v045": max(
                abs(a - b)
                for left, right in zip(source_limits, v045_limits, strict=True)
                for a, b in zip(left, right, strict=True)
            ),
            "source_xml_vs_rcss_mjb": max(
                abs(a - b)
                for left, right in zip(source_limits, mjb_limits, strict=True)
                for a, b in zip(left, right, strict=True)
            ),
            "v045_vs_canonical_contract": maximum_canonical_limit_delta,
        },
        "robot_mass_kg": {
            "source_xml": source["robot_mass_kg"],
            "v045": v045["robot_mass_kg"],
            "rcss_mjb": mjb["robot_mass_kg"],
        },
        "actuator_semantics": {
            "source_xml_and_rcss_mjb": (
                "69 raw control ports: torque, position and velocity for each of "
                "23 canonical joints."
            ),
            "v045": (
                "23 one-to-one joint position servos implementing "
                "kp*(target-q)-kd*qvel, with configured force limits and the "
                "unchanged RCSS joint-level actuator-force caps."
            ),
            "effective_joint_torque_caps": {
                name: float(max(abs(value) for value in cap))
                for name, cap in zip(canonical_names, v045_effective_caps, strict=True)
            },
            "position_servo_requested_caps": v045_actuator_caps_by_joint,
            "rcss_joint_mechanical_caps": {
                name: float(max(abs(value) for value in cap))
                for name, cap in zip(canonical_names, mjb_caps, strict=True)
            },
            "deployment_clip_caps": {
                name: float(limit)
                for name, limit in zip(
                    canonical_names, RCSS_EFFECTIVE_EFFORT_LIMIT, strict=True
                )
            },
            "training_and_deployment_kp": {
                name: float(value)
                for name, value in zip(canonical_names, V045_GETUP_KP, strict=True)
            },
            "training_and_deployment_kd": {
                name: float(value)
                for name, value in zip(canonical_names, V045_GETUP_KD, strict=True)
            },
            "raw_dynamics_identical": False,
            "effective_pd_torque_semantics_audited": True,
        },
        "training_safety_shaping": {
            "ground_contact_sensors": list(training_sensor_names),
            "unsafe_ground_impact_reward_weight": training_cfg.rewards[
                "unsafe_ground_impact"
            ].weight,
            "unsafe_ground_impact_force_limit_n": training_cfg.rewards[
                "unsafe_ground_impact"
            ].params["force_limit_n"],
            "torque_saturation_reward_weight": training_cfg.rewards[
                "torque_saturation"
            ].weight,
            "torque_saturation_fraction": training_cfg.rewards[
                "torque_saturation"
            ].params["fraction"],
            "formal_stable_stand_reward_weight": training_cfg.rewards[
                "formal_stable_stand"
            ].weight,
            "formal_stable_stand_thresholds": training_cfg.rewards[
                "formal_stable_stand"
            ].params,
        },
        "training_reset_contract": {
            "pose_order": ["prone", "supine", "left_side", "right_side"],
            "root_heights_m": list(reset_mdp._ROOT_HEIGHTS_M),
            "parameters": reset_params,
            "uncontrolled_high_drop": False,
            "initial_velocity": "zero",
            "policy_takeover": "immediate",
            "actor_and_critic_nan_policy": "per_term_error",
            "official_source_raw_actor_clip": None,
            "training_raw_actor_clip": None,
            "deployment_raw_actor_clip": None,
            "relative_action_scale_rad": 0.6,
            "maximum_target_delta_per_50hz_step_rad": 0.06,
            "target_limit_margin_rad": 0.08,
            "velocity_brake_lookahead_s": 0.025,
            "final_target_clamp": (
                "intersection of RCSS rule/model limits with dynamic safety margin"
            ),
            "target_is_computed_once_per_50hz_policy_step_and_held_for_four_substeps": True,
            "training_profile": training_profile,
        },
        "source_action_semantics": {
            "checkpoint": source_action_audit["checkpoint"],
            "official_runner_clip_actions": source_action_audit[
                "official_runner_clip_actions"
            ],
            "observed_raw_actions": source_action_audit["observed_raw_actions"],
        },
        "checks": checks,
        "target_model_semantics_passed": semantics_passed,
        "deployment_ready": False,
        "quality_passed": False,
        "conclusion": (
            "Target-model semantics passed. Recovery training and independent RCSS "
            "behavioral evaluation are still required."
            if semantics_passed
            else "Target-model semantics failed; v045 training and promotion are blocked."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical evidence: {output}")
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "target_model_semantics_passed": semantics_passed,
                "deployment_ready": False,
                "failed_checks": [
                    name for name, passed in checks.items() if not passed
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if semantics_passed else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--snapshot-mjb":
        raise SystemExit(_snapshot_mjb(Path(sys.argv[2])))
    raise SystemExit(main())
