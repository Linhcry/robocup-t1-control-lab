#!/usr/bin/env python3
"""CPU dynamics, action guards and real legal/illegal contact probes for v049."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _next_output() -> Path:
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = root / f"v049_cpu_dynamics_smoke_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v049 dynamics smoke filename")


def _finite(obs: dict[str, torch.Tensor]) -> bool:
    return bool(obs) and all(bool(torch.isfinite(value).all()) for value in obs.values())


def _run_case(num_envs: int, seed: int) -> dict:
    from experiments.v049_fullbody_receive import mdp
    from experiments.v049_fullbody_receive.rcss_mjlab_task import (
        rcss_fullbody_receive_env_cfg,
    )
    from locomotion.t1_spec import T1_LOWER_LIMITS, T1_UPPER_LIMITS
    from mjlab.envs import ManagerBasedRlEnv

    cfg = rcss_fullbody_receive_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.terrain.num_envs = num_envs
    cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        shapes = {name: list(value.shape) for name, value in obs.items()}
        observations_finite = _finite(obs)
        rewards_finite = True
        unexpected_done = False
        for _ in range(8):
            obs, reward, terminated, timed_out, _ = env.step(
                torch.zeros((num_envs, 23), device=env.device)
            )
            observations_finite &= _finite(obs)
            rewards_finite &= bool(torch.isfinite(reward).all())
            unexpected_done |= bool((terminated | timed_out).any())

        action = env.action_manager.get_term("joint_pos")
        wrong_shape_rejected = False
        nan_rejected = False
        try:
            action.process_actions(torch.zeros((num_envs, 22), device=env.device))
        except ValueError:
            wrong_shape_rejected = True
        bad = torch.zeros((num_envs, 23), device=env.device)
        bad[0, 0] = float("nan")
        try:
            action.process_actions(bad)
        except FloatingPointError:
            nan_rejected = True
        action.process_actions(torch.full((num_envs, 23), 100.0, device=env.device))
        lower = torch.as_tensor(T1_LOWER_LIMITS, device=env.device).view(1, 23)
        upper = torch.as_tensor(T1_UPPER_LIMITS, device=env.device).view(1, 23)
        targets_clamped = bool(
            ((action.clamped_target >= lower - 1.0e-7) & (action.clamped_target <= upper + 1.0e-7)).all()
        )

        contact_probe = None
        if num_envs == 4:
            env.reset(seed=seed + 1)
            robot: object = env.scene["robot"]
            ball: object = env.scene["ball"]
            left_foot_ids, _ = robot.find_geoms(("left_foot",))
            left_calf_ids, _ = robot.find_geoms(("left_calf",))
            torso_ids, _ = robot.find_geoms(("torso",))
            pose = ball.data.root_link_pose_w.clone()
            pose[0, :3] = robot.data.geom_pos_w[0, left_foot_ids[0]] + torch.tensor(
                [0.0, 0.0, 0.10], device=env.device
            )
            pose[1, :3] = robot.data.geom_pos_w[1, left_calf_ids[0]] + torch.tensor(
                [0.10, 0.0, 0.0], device=env.device
            )
            pose[2, :3] = robot.data.geom_pos_w[2, torso_ids[0]] + torch.tensor(
                [0.16, 0.0, 0.0], device=env.device
            )
            pose[:, 3:7] = 0.0
            pose[:, 3] = 1.0
            velocity = torch.zeros((num_envs, 6), device=env.device)
            velocity[:3, 0] = -1.0
            ball.write_root_link_pose_to_sim(pose)
            ball.write_root_link_velocity_to_sim(velocity)
            state = mdp.receive_state(env)
            for field in (
                "legal_touched",
                "illegal_touched",
                "success",
                "new_success",
            ):
                getattr(state, field)[:] = False
            state.hold_steps[:] = 0
            state.previous_ball_speed[:] = 1.0
            state.last_update_step[:] = -1
            env.sim.forward()
            obs, reward, terminated, timed_out, _ = env.step(
                torch.zeros((num_envs, 23), device=env.device)
            )
            state = mdp.receive_state(env)
            required_hold_steps = round(mdp.HOLD_SECONDS / env.step_dt)
            contact_probe = {
                "foot_contact_latched_from_real_physics": bool(state.legal_touched[0]),
                "shank_contact_latched_from_real_physics": bool(state.legal_touched[1]),
                # The vector environment auto-resets a terminated world in the
                # same call, so its post-step latch already belongs to the new
                # episode.  The returned terminated/not-timeout pair is the
                # immutable evidence for the physical torso collision.
                "torso_contact_detected_as_illegal_termination": bool(terminated[2])
                and not bool(timed_out[2]),
                "single_contact_cannot_fake_one_second_success": not bool(state.success[:3].any()),
                "hold_gate_is_exactly_one_second": required_hold_steps == 50,
                "post_probe_observations_finite": _finite(obs),
                "post_probe_rewards_finite": bool(torch.isfinite(reward).all()),
            }

        checks = {
            "actor_shape": shapes.get("actor") == [num_envs, mdp.ACTOR_OBS_DIM],
            "critic_shape": shapes.get("critic") == [num_envs, mdp.CRITIC_OBS_DIM],
            "observations_finite_for_8_steps": observations_finite,
            "rewards_finite_for_8_steps": rewards_finite,
            "no_unexpected_baseline_termination": not unexpected_done,
            "wrong_action_dimension_rejected": wrong_shape_rejected,
            "nan_action_rejected": nan_rejected,
            "extreme_targets_clamped": targets_clamped,
        }
        if contact_probe:
            checks.update(contact_probe)
        return {
            "num_envs": num_envs,
            "seed": seed,
            "policy_steps": 8,
            "physics_steps": 8 * cfg.decimation,
            "observation_shapes": shapes,
            "contact_probe": contact_probe,
            "checks": checks,
            "passed": all(checks.values()),
        }
    finally:
        env.close()


def main() -> int:
    import mujoco
    import mujoco_warp
    import warp

    cases = [_run_case(1, 49001), _run_case(4, 49004)]
    passed = all(case["passed"] for case in cases)
    output = _next_output()
    sources = {
        str(path.relative_to(EXPERIMENT_ROOT)): _sha256(path)
        for path in sorted(EXPERIMENT_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    payload = {
        "schema_version": "v049_cpu_dynamics_smoke_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "python": sys.version},
        "runtime": {
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "mujoco_warp": getattr(mujoco_warp, "__version__", "unknown"),
            "warp": warp.__version__,
            "device": "cpu",
        },
        "cases": cases,
        "source_sha256": sources,
        "smoke_passed": passed,
        "training_completed": False,
        "fixed_validation_passed": False,
        "heldout_validation_passed": False,
        "quality_passed": False,
        "conclusion": (
            "CPU dynamics and real lower-leg/illegal contact probes passed; no receiving skill was trained."
            if passed
            else "v049 CPU smoke failed; long training is blocked."
        ),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "smoke_passed": passed, "quality_passed": False}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
