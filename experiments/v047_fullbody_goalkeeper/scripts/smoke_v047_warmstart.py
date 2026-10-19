#!/usr/bin/env python3
"""Audit a v045->v047 warm-start and execute one small CPU PPO update."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v047_fullbody_goalkeeper.rcss_goalkeeper_task import (  # noqa: E402
    TASK_ID,
    register_v047_task,
)
from experiments.v047_fullbody_goalkeeper.warmstart import (  # noqa: E402
    SOURCE_ACTOR_DIM,
    SOURCE_BASE_LIN_SLICE,
    TARGET_ACTOR_DIM,
    TARGET_BASE_LIN_SLICE,
    TARGET_CRITIC_DIM,
    sha256,
    transform_source_manifest,
    validate_formal_source,
)


def _unique_output() -> Path:
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        path = root / f"v047_warmstart_cpu_ppo_smoke_run_{run:02d}.json"
        if not path.exists():
            return path
    raise RuntimeError("no unused v047 warm-start smoke filename remains")


def _all_finite(state: dict[str, Any]) -> bool:
    return all(
        not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all())
        for value in state.values()
    )


def _equal_except_first_layer(
    source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> bool:
    return all(
        torch.equal(value, target[key])
        for key, value in source.items()
        if key != "mlp.0.weight"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--warmstart-checkpoint", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--engineering-source-proof", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=47991)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")

    source_path = args.source_checkpoint.resolve()
    target_path = args.warmstart_checkpoint.resolve()
    provenance_path = args.provenance.resolve()
    engineering_proof_path = args.engineering_source_proof.resolve()
    for path in (source_path, target_path, provenance_path, engineering_proof_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    target = torch.load(target_path, map_location="cpu", weights_only=False)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    engineering_proof = json.loads(
        engineering_proof_path.read_text(encoding="utf-8")
    )
    source_actor = source["actor_state_dict"]
    source_critic = source["critic_state_dict"]
    target_actor = target["actor_state_dict"]
    target_critic = target["critic_state_dict"]

    actor_first = target_actor["mlp.0.weight"]
    critic_first = target_critic["mlp.0.weight"]
    new_critic_mask = torch.ones(TARGET_CRITIC_DIM, dtype=torch.bool)
    new_critic_mask[:SOURCE_ACTOR_DIM] = False
    new_critic_mask[TARGET_BASE_LIN_SLICE[0] : TARGET_BASE_LIN_SLICE[1]] = False
    checks: dict[str, bool] = {
        "source_checkpoint_hash_matches_engineering_proof": (
            engineering_proof.get("source_checkpoint", {}).get("sha256")
            == sha256(source_path)
        ),
        "cached_checkpoint_hash_matches_provenance": (
            provenance.get("cached_checkpoint", {}).get("sha256")
            == sha256(target_path)
        ),
        "transform_source_manifest_matches_current_files": (
            provenance.get("transform_source_manifest")
            == transform_source_manifest()
        ),
        "provenance_is_explicitly_non_quality": provenance.get("quality_passed")
        is False,
        "engineering_binding_is_not_formal": provenance.get(
            "formal_source_binding"
        )
        is False,
        "actor_shape_is_512x87": tuple(actor_first.shape)
        == (512, TARGET_ACTOR_DIM),
        "actor_first_75_columns_are_exact": torch.equal(
            actor_first[:, :SOURCE_ACTOR_DIM],
            source_actor["mlp.0.weight"],
        ),
        "actor_new_12_columns_are_zero": bool(
            torch.count_nonzero(actor_first[:, SOURCE_ACTOR_DIM:]) == 0
        ),
        "actor_later_layers_and_distribution_are_exact": _equal_except_first_layer(
            source_actor, target_actor
        ),
        "critic_shape_is_512x107": tuple(critic_first.shape)
        == (512, TARGET_CRITIC_DIM),
        "critic_first_75_columns_are_exact": torch.equal(
            critic_first[:, :SOURCE_ACTOR_DIM],
            source_critic["mlp.0.weight"][:, :SOURCE_ACTOR_DIM],
        ),
        "critic_base_lin_vel_is_remapped_exactly": torch.equal(
            critic_first[
                :, TARGET_BASE_LIN_SLICE[0] : TARGET_BASE_LIN_SLICE[1]
            ],
            source_critic["mlp.0.weight"][
                :, SOURCE_BASE_LIN_SLICE[0] : SOURCE_BASE_LIN_SLICE[1]
            ],
        ),
        "all_new_critic_weight_columns_are_zero": bool(
            torch.count_nonzero(critic_first[:, new_critic_mask]) == 0
        ),
        "critic_later_layers_are_exact": all(
            torch.equal(value, target_critic[key])
            for key, value in source_critic.items()
            if key not in {
                "mlp.0.weight",
                "obs_normalizer._mean",
                "obs_normalizer._var",
                "obs_normalizer._std",
            }
        ),
        "optimizer_state_is_empty": target["optimizer_state_dict"].get("state")
        == {},
        "optimizer_learning_rate_reset": all(
            abs(float(group["lr"]) - 3.0e-4) <= 1.0e-15
            for group in target["optimizer_state_dict"]["param_groups"]
        ),
        "target_iteration_is_zero": target.get("iter") == 0,
        "target_curriculum_counter_is_zero": target.get("infos", {})
        .get("env_state", {})
        .get("common_step_counter")
        == 0,
        "source_and_target_states_are_finite": _all_finite(target_actor)
        and _all_finite(target_critic),
    }
    for key in (
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
    ):
        checks[f"critic_{key}_shared_is_exact"] = torch.equal(
            target_critic[key][:, :SOURCE_ACTOR_DIM],
            source_critic[key][:, :SOURCE_ACTOR_DIM],
        )
        checks[f"critic_{key}_base_lin_is_exact"] = torch.equal(
            target_critic[key][
                :, TARGET_BASE_LIN_SLICE[0] : TARGET_BASE_LIN_SLICE[1]
            ],
            source_critic[key][
                :, SOURCE_BASE_LIN_SLICE[0] : SOURCE_BASE_LIN_SLICE[1]
            ],
        )

    try:
        validate_formal_source(
            source_path, engineering_proof_path, engineering_proof_path
        )
    except (ValueError, FileNotFoundError, TypeError):
        checks["formal_validator_rejects_engineering_proof"] = True
    else:
        checks["formal_validator_rejects_engineering_proof"] = False
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"warm-start mapping failed before PPO smoke: {failed}")

    register_v047_task()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(TASK_ID, play=False)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain.num_envs = 1
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.num_steps_per_env = args.steps
    agent_cfg.max_iterations = 1
    agent_cfg.algorithm.num_learning_epochs = 1
    agent_cfg.algorithm.num_mini_batches = 2
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device="cpu")
    try:
        loaded_infos = runner.load(str(target_path), strict=True, map_location="cpu")
        actor_before = {
            name: value.detach().clone()
            for name, value in runner.alg.actor.named_parameters()
        }
        critic_before = {
            name: value.detach().clone()
            for name, value in runner.alg.critic.named_parameters()
        }
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        actor_after = dict(runner.alg.actor.named_parameters())
        critic_after = dict(runner.alg.critic.named_parameters())
        actor_changed = any(
            not torch.equal(value, actor_after[name].detach())
            for name, value in actor_before.items()
        )
        critic_changed = any(
            not torch.equal(value, critic_after[name].detach())
            for name, value in critic_before.items()
        )
        post_update_finite = all(
            bool(torch.isfinite(value).all())
            for value in list(actor_after.values()) + list(critic_after.values())
        )
        policy_steps_after_update = int(env.unwrapped.common_step_counter)
        checks.update(
            {
                "strict_runner_load_succeeded": True,
                "loaded_curriculum_counter_is_zero": loaded_infos["env_state"][
                    "common_step_counter"
                ]
                == 0,
                # rsl-rl stores the zero-based index of the last completed
                # iteration, so one update starting from iteration zero leaves
                # current_learning_iteration at 0 rather than 1.
                "rsl_last_completed_iteration_index_is_zero": (
                    runner.current_learning_iteration == 0
                ),
                "one_cpu_ppo_update_completed": (
                    policy_steps_after_update >= args.steps
                    and actor_changed
                    and critic_changed
                ),
                "actor_parameters_changed_after_update": actor_changed,
                "critic_parameters_changed_after_update": critic_changed,
                "post_update_parameters_are_finite": post_update_finite,
            }
        )
    finally:
        env.close()

    passed = bool(checks) and all(checks.values())
    output = _unique_output()
    payload = {
        "schema_version": "v047_v045_warmstart_cpu_ppo_smoke_v1",
        "task_id": TASK_ID,
        "source_checkpoint": {"path": str(source_path), "sha256": sha256(source_path)},
        "warmstart_checkpoint": {"path": str(target_path), "sha256": sha256(target_path)},
        "provenance": {"path": str(provenance_path), "sha256": sha256(provenance_path)},
        "engineering_source_proof": {
            "path": str(engineering_proof_path),
            "sha256": sha256(engineering_proof_path),
        },
        "source_manifest": {
            "warmstart.py": sha256(EXPERIMENT_ROOT / "warmstart.py"),
            "rcss_goalkeeper_task.py": sha256(
                EXPERIMENT_ROOT / "rcss_goalkeeper_task.py"
            ),
            "scripts/train_v047.py": sha256(
                EXPERIMENT_ROOT / "scripts" / "train_v047.py"
            ),
            "scripts/smoke_v047_warmstart.py": sha256(Path(__file__).resolve()),
        },
        "device": "cpu",
        "num_environments": 1,
        "rollout_steps": args.steps,
        "ppo_learning_epochs": 1,
        "ppo_mini_batches": 2,
        "checks": checks,
        "smoke_passed": passed,
        "formal_v045_source_proof_available": False,
        "gpu_training_completed": False,
        "fixed_validation_completed": False,
        "held_out_test_completed": False,
        "quality_passed": False,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "smoke_passed": passed,
                "failed_checks": [name for name, ok in checks.items() if not ok],
                "quality_passed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
