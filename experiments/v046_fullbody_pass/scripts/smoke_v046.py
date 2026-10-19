#!/usr/bin/env python3
"""Run deterministic 1/4-environment CPU dynamics and contact probes."""

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
    metrics = EXPERIMENT_ROOT / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = metrics / f"v046_cpu_dynamics_smoke_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v046 dynamics smoke filename")


def _finite_observations(obs: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(value).all().item() for value in obs.values())


def _run_case(num_envs: int, seed: int) -> dict:
    from experiments.v046_fullbody_pass.mdp.commands import DirectionalPassCommand
    from experiments.v046_fullbody_pass.rcss_mjlab_task import (
        rcss_t1_directional_pass_env_cfg,
    )
    from locomotion.t1_spec import T1_LOWER_LIMITS, T1_UPPER_LIMITS
    from mjlab.envs import ManagerBasedRlEnv

    cfg = rcss_t1_directional_pass_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.terrain.num_envs = num_envs
    cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        initial_shapes = {name: list(value.shape) for name, value in obs.items()}
        all_finite = _finite_observations(obs)
        reward_finite = True
        unexpected_termination = False
        steps = 8
        for _ in range(steps):
            obs, reward, terminated, timed_out, _ = env.step(
                torch.zeros((num_envs, 23), device=env.device)
            )
            all_finite &= _finite_observations(obs)
            reward_finite &= torch.isfinite(reward).all().item()
            unexpected_termination |= (terminated | timed_out).any().item()

        action_term = env.action_manager.get_term("joint_pos")
        wrong_shape_rejected = False
        nan_rejected = False
        try:
            action_term.process_actions(torch.zeros((num_envs, 22), device=env.device))
        except ValueError:
            wrong_shape_rejected = True
        invalid = torch.zeros((num_envs, 23), device=env.device)
        invalid[0, 0] = float("nan")
        try:
            action_term.process_actions(invalid)
        except FloatingPointError:
            nan_rejected = True

        extreme = torch.full((num_envs, 23), 100.0, device=env.device)
        action_term.process_actions(extreme)
        lower = torch.as_tensor(T1_LOWER_LIMITS, device=env.device).view(1, 23)
        upper = torch.as_tensor(T1_UPPER_LIMITS, device=env.device).view(1, 23)
        targets_are_clamped = bool(
            (
                (action_term.clamped_target >= lower - 1.0e-7)
                & (action_term.clamped_target <= upper + 1.0e-7)
            )
            .all()
            .item()
        )

        contact_probe = None
        if num_envs == 4:
            # Probe 0 places the ball just above the left foot. Probe 1 places
            # it against the torso. These are physical overlaps followed by a
            # real MuJoCo step, not manually edited sensor bits.
            env.reset(seed=seed + 1)
            robot = env.scene["robot"]
            ball = env.scene["ball"]
            command = env.command_manager.get_term("directional_pass")
            assert isinstance(command, DirectionalPassCommand)
            left_foot_ids, _ = robot.find_geoms(("left_foot",))
            torso_ids, _ = robot.find_geoms(("torso",))
            left_foot_pos = robot.data.geom_pos_w[0, left_foot_ids[0]].clone()
            torso_pos = robot.data.geom_pos_w[1, torso_ids[0]].clone()
            pose = ball.data.root_link_pose_w.clone()
            pose[0, :3] = left_foot_pos + torch.tensor(
                [0.0, 0.0, 0.10], device=env.device
            )
            pose[1, :3] = torso_pos + torch.tensor(
                [0.16, 0.0, 0.0], device=env.device
            )
            pose[:, 3:7] = 0.0
            pose[:, 3] = 1.0
            velocity = torch.zeros((num_envs, 6), device=env.device)
            ball.write_root_link_pose_to_sim(pose)
            ball.write_root_link_velocity_to_sim(velocity)
            command.foot_touched[:] = False
            command.upper_body_touched[:] = False
            env.sim.forward()
            obs, reward, terminated, timed_out, _ = env.step(
                torch.zeros((num_envs, 23), device=env.device)
            )
            contact_probe = {
                "foot_contact_latched_from_physics": bool(
                    command.foot_touched[0].item()
                ),
                "upper_body_probe_terminated": bool(terminated[1].item()),
                "upper_body_probe_not_timeout": not bool(timed_out[1].item()),
                "post_probe_observations_finite": _finite_observations(obs),
                "post_probe_rewards_finite": bool(torch.isfinite(reward).all().item()),
            }

        checks = {
            "actor_shape_is_num_envs_by_84": initial_shapes.get("actor")
            == [num_envs, 84],
            "critic_shape_is_num_envs_by_101": initial_shapes.get("critic")
            == [num_envs, 101],
            "observations_finite_for_8_steps": all_finite,
            "rewards_finite_for_8_steps": reward_finite,
            "no_unexpected_termination_in_8_step_baseline": not unexpected_termination,
            "wrong_action_dimension_rejected": wrong_shape_rejected,
            "nan_action_rejected": nan_rejected,
            "extreme_targets_clamped_to_t1_contract": targets_are_clamped,
        }
        if contact_probe is not None:
            checks.update(contact_probe)
        return {
            "num_envs": num_envs,
            "seed": seed,
            "physics_steps": steps * cfg.decimation,
            "policy_steps": steps,
            "observation_shapes": initial_shapes,
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

    output = _next_output()
    cases = [_run_case(1, 46001), _run_case(4, 46004)]
    smoke_passed = all(case["passed"] for case in cases)
    sources = {
        str(path.relative_to(EXPERIMENT_ROOT)): _sha256(path)
        for path in sorted(EXPERIMENT_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    payload = {
        "schema_version": "v046_cpu_dynamics_smoke_v1",
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
        "smoke_passed": smoke_passed,
        "training_completed": False,
        "validation_passed": False,
        "quality_passed": False,
        "receive_stage_completed": False,
        "conclusion": (
            "The 1/4-environment CPU dynamics, action guards and real contact "
            "probes passed. No passing or receiving skill has been learned yet."
            if smoke_passed
            else "v046 CPU smoke failed; training is blocked."
        ),
    }
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "smoke_passed": smoke_passed,
                "quality_passed": False,
                "failed_checks": {
                    str(case["num_envs"]): [
                        name for name, value in case["checks"].items() if not value
                    ]
                    for case in cases
                },
            },
            ensure_ascii=False,
        )
    )
    return 0 if smoke_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

