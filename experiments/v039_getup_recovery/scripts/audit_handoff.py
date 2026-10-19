"""Prove whether the upright-to-stand/walk endpoint is currently controllable."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v039_getup_recovery.artifacts import write_json  # noqa: E402
from experiments.v039_getup_recovery.getup_env import T1GetUpEnv  # noqa: E402
from locomotion.official_t1_policy import (  # noqa: E402
    MotionCommand,
    OfficialT1LocomotionPolicy,
)
from locomotion.rcss_t1_controller import PREPARE_QPOS  # noqa: E402
from locomotion.t1_spec import T1_LOWER_LIMITS, T1_UPPER_LIMITS  # noqa: E402


OFFICIAL_POLICY = (
    PROJECT_ROOT.parents[1] / "third_party" / "booster_gym" / "deploy" / "models" / "T1.pt"
)


def _upright_reset(env: T1GetUpEnv) -> None:
    env.reset(seed=39900)
    env.data.qpos[:3] = np.asarray([0.0, 0.0, 0.66])
    env.data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    env.data.qpos[env.qpos_adrs] = PREPARE_QPOS
    env.data.qvel[:] = 0.0
    env.filtered_target = PREPARE_QPOS.astype(np.float64)
    env.previous_action.fill(0.0)
    mujoco.mj_forward(env.model, env.data)


def _sample(env: T1GetUpEnv) -> dict[str, float | bool]:
    _, linear_velocity, _, height, upright = env._root_features()
    qpos, _ = env._joint_state()
    overshoot = max(
        float(np.max(np.maximum(T1_LOWER_LIMITS - qpos, 0.0))),
        float(np.max(np.maximum(qpos - T1_UPPER_LIMITS, 0.0))),
    )
    return {
        "height_m": height,
        "upright_cos": upright,
        "root_speed_mps": float(np.linalg.norm(linear_velocity)),
        "joint_overshoot_rad": overshoot,
        "stable": bool(height >= 0.60 and upright >= 0.90),
    }


def _physics_step(env: T1GetUpEnv, action: np.ndarray) -> dict[str, float | bool]:
    env._apply_action(action)
    for _ in range(env.decimation):
        mujoco.mj_step(env.model, env.data)
    return _sample(env)


def _run_static_prepare() -> dict[str, object]:
    env = T1GetUpEnv(stage=0)
    _upright_reset(env)
    samples = []
    failure_time = None
    for step in range(int(round(3.0 / (env.model.opt.timestep * env.decimation)))):
        sample = _physics_step(env, np.zeros(23, dtype=np.float32))
        if step % 10 == 0:
            samples.append({"time_s": (step + 1) * 0.02, **sample})
        if step >= 10 and not sample["stable"]:
            failure_time = (step + 1) * 0.02
            break
    final = _sample(env)
    env.close()
    return {
        "controller": "static_prepare_pose_PD",
        "stand_3s_passed": failure_time is None,
        "failure_time_s": failure_time,
        "final": final,
        "samples": samples,
    }


def _run_official_policy() -> dict[str, object]:
    env = T1GetUpEnv(stage=0)
    _upright_reset(env)
    if not OFFICIAL_POLICY.exists():
        env.close()
        return {
            "controller": "official_T1_pt",
            "stand_3s_passed": False,
            "walk_5s_passed": False,
            "blocker": f"Missing {OFFICIAL_POLICY}",
        }
    policy = OfficialT1LocomotionPolicy(OFFICIAL_POLICY, device="cpu")
    policy.reset()
    samples = []
    failure_phase = None
    failure_time = None
    total_steps = int(round(8.0 / (env.model.opt.timestep * env.decimation)))
    for step in range(total_steps):
        time_now = step * env.model.opt.timestep * env.decimation
        projected_gravity, _, angular_velocity, _, _ = env._root_features()
        qpos, qvel = env._joint_state()
        if time_now < 3.0:
            command = MotionCommand()
            phase = "stand"
        else:
            command = MotionCommand(vx=0.22, vy=0.10, yaw_rate=0.10)
            phase = "walk"
        target = policy.infer(
            time_now,
            qpos.astype(np.float32),
            qvel.astype(np.float32),
            angular_velocity.astype(np.float32),
            projected_gravity.astype(np.float32),
            command,
        )
        action = np.clip((target - PREPARE_QPOS) / env.target_scale, -1.0, 1.0)
        sample = _physics_step(env, action.astype(np.float32))
        if step % 10 == 0:
            samples.append({"time_s": time_now + 0.02, "phase": phase, **sample})
        if step >= 10 and not sample["stable"]:
            failure_phase = phase
            failure_time = time_now + 0.02
            break
    final = _sample(env)
    env.close()
    return {
        "controller": "official_T1_pt_12leg_plus_fullbody_PD",
        "stand_3s_passed": failure_phase is None or failure_phase == "walk",
        "walk_5s_passed": failure_phase is None and total_steps * 0.02 >= 8.0,
        "failure_phase": failure_phase,
        "failure_time_s": failure_time,
        "final": final,
        "samples": samples,
    }


def main() -> None:
    static_result = _run_static_prepare()
    official_result = _run_official_policy()
    payload = {
        "experiment": "v039_getup_recovery",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": "upright start -> stand 3 s -> walk 5 s",
        "static_prepare": static_result,
        "official_locomotion_policy": official_result,
        "quality_passed": bool(
            official_result["stand_3s_passed"]
            and official_result["walk_5s_passed"]
        ),
        "conclusion": (
            "No validated full-body handoff controller is available; a get-up policy cannot be promoted."
        ),
    }
    output = write_json(
        Path(__file__).resolve().parents[1] / "metrics" / "v039_stand_walk_handoff_audit.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved={output}")
    if not payload["quality_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
