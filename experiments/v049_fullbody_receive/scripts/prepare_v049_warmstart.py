#!/usr/bin/env python3
"""Create a shape-aware, hash-bound v045-to-v049 network warm-start."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v045_rcss_t1_getup_adaptation.source_approval import (  # noqa: E402
    validate_source_approval,
)


SOURCE_ACTOR_DIM = 75
SOURCE_CRITIC_DIM = 78
TARGET_ACTOR_DIM = 81
TARGET_CRITIC_DIM = 109
COMMON_DIM = 75
SOURCE_BASE_LIN_SLICE = (75, 78)
TARGET_BASE_LIN_SLICE = (81, 84)
TARGET_LEARNING_RATE = 3.0e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _checkpoint_ref_matches(reference: Any, checkpoint: Path, checkpoint_sha: str) -> bool:
    if not isinstance(reference, dict):
        return False
    path = reference.get("path")
    return bool(
        isinstance(path, str)
        and path
        and Path(path).expanduser().resolve() == checkpoint
        and reference.get("sha256") == checkpoint_sha
    )


def _proof_ref_matches(reference: Any, proof: Path, proof_sha: str) -> bool:
    if not isinstance(reference, dict):
        return False
    path = reference.get("path")
    return bool(
        isinstance(path, str)
        and path
        and Path(path).expanduser().resolve() == proof
        and reference.get("sha256") == proof_sha
    )


def validate_source_evidence(
    source_checkpoint: Path,
    training_summary_path: Path | None,
    candidate_proof_path: Path | None,
    *,
    engineering_smoke: bool,
    source_approval_path: Path | None = None,
) -> dict[str, Any]:
    """Use legacy evidence only for smoke; formal mode requires terminal approval."""

    source_checkpoint = source_checkpoint.expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    source_sha = sha256(source_checkpoint)

    if engineering_smoke:
        if source_approval_path is not None:
            raise ValueError("engineering smoke cannot consume a formal source approval")
        if training_summary_path is None or candidate_proof_path is None:
            raise ValueError("engineering smoke requires its non-promotable summary and proof")
        training_summary_path = training_summary_path.expanduser().resolve()
        candidate_proof_path = candidate_proof_path.expanduser().resolve()
        summary = load_json(training_summary_path, "v045 engineering source summary")
        candidate = load_json(candidate_proof_path, "v045 engineering candidate proof")
        summary_sha = sha256(training_summary_path)
        candidate_sha = sha256(candidate_proof_path)
        if summary.get("schema_version") != "v049_v045_engineering_source_summary_v1":
            raise ValueError("engineering summary schema mismatch")
        if summary.get("formal_protocol") is not False:
            raise ValueError("engineering summary must explicitly be non-formal")
        if summary.get("training_completed") is not False or summary.get("quality_passed") is not False:
            raise ValueError("engineering summary must not claim completed v045 training")
        if not _checkpoint_ref_matches(summary.get("source_checkpoint"), source_checkpoint, source_sha):
            raise ValueError("engineering summary does not bind the source checkpoint")
        if not _proof_ref_matches(summary.get("candidate_proof"), candidate_proof_path, candidate_sha):
            raise ValueError("engineering summary does not bind the candidate proof")
        if candidate.get("schema_version") != "v045_model800_safety_4096_probe_v1":
            raise ValueError("engineering candidate proof schema mismatch")
        if candidate.get("ppo_update_completed") is not True:
            raise ValueError("engineering candidate did not complete its v045 PPO probe")
        if candidate.get("training_completed") is not False or candidate.get("quality_passed") is not False:
            raise ValueError("engineering candidate must remain non-promotable")
        if not _checkpoint_ref_matches(candidate.get("output_checkpoint"), source_checkpoint, source_sha):
            raise ValueError("engineering candidate output does not match source checkpoint")
        audit_ref = summary.get("v045_target_model_audit")
        if not isinstance(audit_ref, dict) or not isinstance(audit_ref.get("path"), str):
            raise ValueError("engineering summary has no v045 target-model audit")
        audit_path = Path(audit_ref["path"]).expanduser().resolve()
        audit = load_json(audit_path, "v045 target-model audit")
        if audit_ref.get("sha256") != sha256(audit_path):
            raise ValueError("v045 target-model audit hash mismatch")
        if (
            audit.get("schema_version") != "v045_target_model_semantic_audit_v3"
            or audit.get("target_model_semantics_passed") is not True
            or audit.get("quality_passed") is not False
        ):
            raise ValueError("v045 target-model semantics are not a valid engineering source")
        return {
            "source_checkpoint": {
                "path": str(source_checkpoint),
                "sha256": source_sha,
            },
            "v045_training_summary": {
                "path": str(training_summary_path),
                "sha256": summary_sha,
                "schema_version": summary.get("schema_version"),
            },
            "v045_candidate_proof": {
                "path": str(candidate_proof_path),
                "sha256": candidate_sha,
                "schema_version": candidate.get("schema_version"),
            },
            "formal_source_approved": False,
            "promotable_source_binding": False,
            "engineering_smoke_only": True,
        }
    else:
        if training_summary_path is not None or candidate_proof_path is not None:
            raise ValueError(
                "formal mode accepts only the terminal source approval, not shortlist evidence"
            )
        if source_approval_path is None:
            raise ValueError("formal mode requires a v045 downstream source approval")
        binding = validate_source_approval(source_checkpoint, source_approval_path)
        if binding.get("formal_source_approved") is not True:
            raise ValueError("v045 downstream source approval did not authorize transfer")
        return {**binding, "engineering_smoke_only": False}


def _layout_for_env(env: Any) -> dict[str, Any]:
    manager = env.observation_manager
    result: dict[str, Any] = {}
    for group in ("actor", "critic"):
        offset = 0
        terms = []
        names = manager.active_terms[group]
        dims = manager.group_obs_term_dim[group]
        for name, shape in zip(names, dims, strict=True):
            size = 1
            for value in shape:
                size *= int(value)
            terms.append(
                {"name": name, "shape": list(shape), "slice": [offset, offset + size]}
            )
            offset += size
        result[group] = {
            "shape": list(manager.group_obs_dim[group]),
            "terms": terms,
            "total_dim": offset,
        }
    return result


def inspect_live_observation_contract() -> dict[str, Any]:
    """Construct both real CPU environments and derive slices from managers."""

    from mjlab.envs import ManagerBasedRlEnv
    from experiments.v045_rcss_t1_getup_adaptation.rcss_mjlab_task import (
        rcss_t1_getup_env_cfg,
    )
    from experiments.v049_fullbody_receive.rcss_mjlab_task import (
        rcss_fullbody_receive_env_cfg,
    )

    layouts: dict[str, Any] = {}
    configs = {
        "v045": rcss_t1_getup_env_cfg(),
        "v049": rcss_fullbody_receive_env_cfg(),
    }
    for name, cfg in configs.items():
        cfg.scene.num_envs = 1
        cfg.scene.terrain.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        try:
            layouts[name] = _layout_for_env(env)
        finally:
            env.close()

    expected = {
        "v045": {
            "actor": [
                ("base_ang_vel", [0, 3]),
                ("projected_gravity", [3, 6]),
                ("joint_pos", [6, 29]),
                ("joint_vel", [29, 52]),
                ("actions", [52, 75]),
            ],
            "critic": [
                ("base_ang_vel", [0, 3]),
                ("projected_gravity", [3, 6]),
                ("joint_pos", [6, 29]),
                ("joint_vel", [29, 52]),
                ("actions", [52, 75]),
                ("base_lin_vel", [75, 78]),
            ],
        },
        "v049": {
            "actor": [
                ("base_ang_vel", [0, 3]),
                ("projected_gravity", [3, 6]),
                ("joint_pos", [6, 29]),
                ("joint_vel", [29, 52]),
                ("last_action", [52, 75]),
                ("ball_relative_position", [75, 78]),
                ("ball_relative_velocity", [78, 81]),
            ],
            "critic": [
                ("base_ang_vel", [0, 3]),
                ("projected_gravity", [3, 6]),
                ("joint_pos", [6, 29]),
                ("joint_vel", [29, 52]),
                ("last_action", [52, 75]),
                ("ball_relative_position", [75, 78]),
                ("ball_relative_velocity", [78, 81]),
                ("base_linear_velocity_privileged", [81, 84]),
                ("ball_world_state_privileged", [84, 93]),
                ("scenario_privileged", [93, 99]),
                ("receive_state_privileged", [99, 109]),
            ],
        },
    }
    for version, groups in expected.items():
        for group, expected_terms in groups.items():
            actual = [
                (term["name"], term["slice"])
                for term in layouts[version][group]["terms"]
            ]
            if actual != expected_terms:
                raise ValueError(
                    f"{version} {group} observation layout drifted: "
                    f"expected={expected_terms}, actual={actual}"
                )

    # Confirm the common 75-D semantics from callable configuration, not names
    # alone. ``actions`` and ``last_action`` are an intentional naming alias.
    v045_cfg = configs["v045"]
    v049_cfg = configs["v049"]
    for group in ("actor", "critic"):
        source_terms = list(v045_cfg.observations[group].terms.values())[:5]
        target_terms = list(v049_cfg.observations[group].terms.values())[:5]
        source_funcs = [getattr(term.func, "__name__", "") for term in source_terms]
        target_funcs = [getattr(term.func, "__name__", "") for term in target_terms]
        if source_funcs != target_funcs:
            raise ValueError(f"{group} common observation callables differ")
    if v045_cfg.observations["critic"].terms["joint_pos"].params.get("biased", False):
        raise ValueError("v045 critic joint position unexpectedly includes encoder bias")
    if v049_cfg.observations["critic"].terms["joint_pos"].params.get("biased", False):
        raise ValueError("v049 critic joint position must match v045 exact semantics")
    return layouts


def _require_checkpoint_payload(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} checkpoint is not a dictionary")
    required = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"{label} checkpoint is missing {missing}")
    return payload


def _neutral_normalizer(state: dict[str, torch.Tensor], prefix: str, dim: int) -> None:
    mean = state[f"{prefix}._mean"]
    var = state[f"{prefix}._var"]
    std = state[f"{prefix}._std"]
    count = state[f"{prefix}.count"]
    if mean.shape != (1, dim) or var.shape != (1, dim) or std.shape != (1, dim):
        raise ValueError(f"target {prefix} shape does not match {dim}")
    mean.zero_()
    var.fill_(1.0)
    std.fill_(1.0)
    count.zero_()


def transform_checkpoint(
    source_path: Path,
    target_template_path: Path,
    output_path: Path,
    evidence: dict[str, Any],
    layouts: dict[str, Any],
) -> dict[str, Any]:
    source_path = source_path.resolve()
    target_template_path = target_template_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    if not target_template_path.is_file():
        raise FileNotFoundError(target_template_path)
    source = _require_checkpoint_payload(
        torch.load(source_path, map_location="cpu", weights_only=False), "v045 source"
    )
    target = copy.deepcopy(
        _require_checkpoint_payload(
            torch.load(target_template_path, map_location="cpu", weights_only=False),
            "v049 target template",
        )
    )
    source_actor = source["actor_state_dict"]
    source_critic = source["critic_state_dict"]
    target_actor = target["actor_state_dict"]
    target_critic = target["critic_state_dict"]

    expected_source_actor_keys = {
        "distribution.log_std_param",
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
        "mlp.4.weight",
        "mlp.4.bias",
        "mlp.6.weight",
        "mlp.6.bias",
    }
    if set(source_actor) != expected_source_actor_keys:
        raise ValueError(f"unexpected v045 actor keys: {sorted(source_actor)}")
    expected_target_actor_keys = expected_source_actor_keys | {
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "obs_normalizer.count",
    }
    if set(target_actor) != expected_target_actor_keys:
        raise ValueError(f"unexpected v049 actor keys: {sorted(target_actor)}")
    normalizer_keys = {
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "obs_normalizer.count",
    }
    expected_source_critic_keys = {
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
        "mlp.4.weight",
        "mlp.4.bias",
        "mlp.6.weight",
        "mlp.6.bias",
    } | normalizer_keys
    if set(source_critic) != expected_source_critic_keys:
        raise ValueError(f"unexpected v045 critic keys: {sorted(source_critic)}")
    if set(target_critic) != expected_source_critic_keys:
        raise ValueError(f"unexpected v049 critic keys: {sorted(target_critic)}")

    if source_actor["mlp.0.weight"].shape != (512, SOURCE_ACTOR_DIM):
        raise ValueError("v045 actor first layer is not 512x75")
    if target_actor["mlp.0.weight"].shape != (512, TARGET_ACTOR_DIM):
        raise ValueError("v049 actor first layer is not 512x81")
    if source_critic["mlp.0.weight"].shape != (512, SOURCE_CRITIC_DIM):
        raise ValueError("v045 critic first layer is not 512x78")
    if target_critic["mlp.0.weight"].shape != (512, TARGET_CRITIC_DIM):
        raise ValueError("v049 critic first layer is not 512x109")

    actor_first = torch.zeros_like(target_actor["mlp.0.weight"])
    actor_first[:, :COMMON_DIM].copy_(source_actor["mlp.0.weight"])
    target_actor["mlp.0.weight"] = actor_first
    for key in sorted(expected_source_actor_keys - {"mlp.0.weight"}):
        if target_actor[key].shape != source_actor[key].shape:
            raise ValueError(f"incompatible actor tensor {key}")
        target_actor[key] = source_actor[key].clone()
    _neutral_normalizer(target_actor, "obs_normalizer", TARGET_ACTOR_DIM)

    critic_first = torch.zeros_like(target_critic["mlp.0.weight"])
    critic_first[:, :COMMON_DIM].copy_(source_critic["mlp.0.weight"][:, :COMMON_DIM])
    critic_first[:, TARGET_BASE_LIN_SLICE[0] : TARGET_BASE_LIN_SLICE[1]].copy_(
        source_critic["mlp.0.weight"][:, SOURCE_BASE_LIN_SLICE[0] : SOURCE_BASE_LIN_SLICE[1]]
    )
    target_critic["mlp.0.weight"] = critic_first
    for key in sorted(
        expected_source_critic_keys - normalizer_keys - {"mlp.0.weight"}
    ):
        if target_critic[key].shape != source_critic[key].shape:
            raise ValueError(f"incompatible critic tensor {key}")
        target_critic[key] = source_critic[key].clone()

    for suffix in ("_mean", "_var", "_std"):
        source_value = source_critic[f"obs_normalizer.{suffix}"]
        target_value = target_critic[f"obs_normalizer.{suffix}"]
        if source_value.shape != (1, SOURCE_CRITIC_DIM) or target_value.shape != (
            1,
            TARGET_CRITIC_DIM,
        ):
            raise ValueError(f"critic normalizer {suffix} shape mismatch")
        target_value.fill_(0.0 if suffix == "_mean" else 1.0)
        target_value[:, :COMMON_DIM].copy_(source_value[:, :COMMON_DIM])
        target_value[
            :, TARGET_BASE_LIN_SLICE[0] : TARGET_BASE_LIN_SLICE[1]
        ].copy_(source_value[:, SOURCE_BASE_LIN_SLICE[0] : SOURCE_BASE_LIN_SLICE[1]])
    target_critic["obs_normalizer.count"] = source_critic[
        "obs_normalizer.count"
    ].clone()

    optimizer = copy.deepcopy(target["optimizer_state_dict"])
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = TARGET_LEARNING_RATE
    target["optimizer_state_dict"] = optimizer
    target["iter"] = 0
    target["infos"] = {
        "env_state": {"common_step_counter": 0},
        "warm_start_provenance": {
            **evidence,
            "target_template": {
                "path": str(target_template_path),
                "sha256": sha256(target_template_path),
                "used_for_shapes_and_target_optimizer_parameter_ids_only": True,
            },
            "mapping": {
                "actor_common": [0, 75],
                "actor_new_zero": [75, 81],
                "critic_common": {"source": [0, 75], "target": [0, 75]},
                "critic_base_lin": {"source": [75, 78], "target": [81, 84]},
                "critic_new_zero": [[75, 81], [84, 109]],
            },
            "optimizer_state_reset": True,
            "iteration_reset": True,
            "curriculum_counter_reset": True,
            "learning_rate_reset_to": TARGET_LEARNING_RATE,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target, output_path)

    loaded = _require_checkpoint_payload(
        torch.load(output_path, map_location="cpu", weights_only=False), "v049 output"
    )
    actor = loaded["actor_state_dict"]
    critic = loaded["critic_state_dict"]
    actor_compatible = expected_source_actor_keys - {"mlp.0.weight"}
    critic_compatible = expected_source_critic_keys - normalizer_keys - {
        "mlp.0.weight"
    }
    checks = {
        "live_v045_actor_dim_is_75": layouts["v045"]["actor"]["total_dim"] == 75,
        "live_v045_critic_dim_is_78": layouts["v045"]["critic"]["total_dim"] == 78,
        "live_v049_actor_dim_is_81": layouts["v049"]["actor"]["total_dim"] == 81,
        "live_v049_critic_dim_is_109": layouts["v049"]["critic"]["total_dim"] == 109,
        "actor_first_75_columns_copied_exactly": torch.equal(
            actor["mlp.0.weight"][:, :75], source_actor["mlp.0.weight"]
        ),
        "actor_new_6_columns_are_zero": bool(
            torch.count_nonzero(actor["mlp.0.weight"][:, 75:81]) == 0
        ),
        "critic_common_75_columns_copied_exactly": torch.equal(
            critic["mlp.0.weight"][:, :75], source_critic["mlp.0.weight"][:, :75]
        ),
        "critic_base_lin_75_78_mapped_to_81_84": torch.equal(
            critic["mlp.0.weight"][:, 81:84], source_critic["mlp.0.weight"][:, 75:78]
        ),
        "critic_other_new_columns_are_zero": bool(
            torch.count_nonzero(critic["mlp.0.weight"][:, 75:81]) == 0
            and torch.count_nonzero(critic["mlp.0.weight"][:, 84:109]) == 0
        ),
        "all_other_actor_layers_and_distribution_copied": all(
            torch.equal(actor[key], source_actor[key]) for key in actor_compatible
        ),
        "all_other_critic_layers_copied": all(
            torch.equal(critic[key], source_critic[key]) for key in critic_compatible
        ),
        "critic_common_normalizer_stats_copied": all(
            torch.equal(
                critic[f"obs_normalizer.{suffix}"][:, :75],
                source_critic[f"obs_normalizer.{suffix}"][:, :75],
            )
            for suffix in ("_mean", "_var", "_std")
        ),
        "critic_base_lin_normalizer_stats_remapped": all(
            torch.equal(
                critic[f"obs_normalizer.{suffix}"][:, 81:84],
                source_critic[f"obs_normalizer.{suffix}"][:, 75:78],
            )
            for suffix in ("_mean", "_var", "_std")
        ),
        "new_normalizer_dimensions_are_neutral": bool(
            torch.count_nonzero(actor["obs_normalizer._mean"]) == 0
            and torch.all(actor["obs_normalizer._var"] == 1)
            and torch.all(actor["obs_normalizer._std"] == 1)
            and int(actor["obs_normalizer.count"]) == 0
            and torch.count_nonzero(critic["obs_normalizer._mean"][:, 75:81]) == 0
            and torch.count_nonzero(critic["obs_normalizer._mean"][:, 84:109]) == 0
            and torch.all(critic["obs_normalizer._var"][:, 75:81] == 1)
            and torch.all(critic["obs_normalizer._var"][:, 84:109] == 1)
            and torch.all(critic["obs_normalizer._std"][:, 75:81] == 1)
            and torch.all(critic["obs_normalizer._std"][:, 84:109] == 1)
        ),
        "optimizer_state_empty": len(loaded["optimizer_state_dict"]["state"]) == 0,
        "learning_rate_reset": all(
            group["lr"] == TARGET_LEARNING_RATE
            for group in loaded["optimizer_state_dict"]["param_groups"]
        ),
        "iteration_reset": int(loaded["iter"]) == 0,
        "curriculum_counter_reset": loaded["infos"]["env_state"][
            "common_step_counter"
        ]
        == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "warm-start post-save verification failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    return {
        "output_checkpoint": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "size_bytes": output_path.stat().st_size,
        },
        "target_template": {
            "path": str(target_template_path),
            "sha256": sha256(target_template_path),
        },
        "observation_layouts": layouts,
        "mapping_checks": checks,
    }


def _next_output_paths(output_dir: Path, engineering_smoke: bool) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "engineering" if engineering_smoke else "formal"
    for run in range(1, 10_000):
        stem = f"v049_v045_warmstart_{mode}_run_{run:02d}"
        checkpoint = output_dir / f"{stem}.pt"
        proof = output_dir / f"{stem}.provenance.json"
        if not checkpoint.exists() and not proof.exists():
            return checkpoint, proof
    raise RuntimeError("no unused v049 warm-start artifact filename")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-template", type=Path, required=True)
    parser.add_argument("--v045-source-approval", type=Path)
    parser.add_argument("--v045-training-summary", type=Path)
    parser.add_argument("--v045-candidate-proof", type=Path)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "artifacts")
    args = parser.parse_args()
    if args.engineering_smoke:
        if args.v045_source_approval is not None:
            parser.error("engineering smoke cannot consume --v045-source-approval")
        if args.v045_training_summary is None or args.v045_candidate_proof is None:
            parser.error(
                "engineering smoke requires --v045-training-summary and --v045-candidate-proof"
            )
    else:
        if args.v045_source_approval is None:
            parser.error("formal warm-start requires --v045-source-approval")
        if args.v045_training_summary is not None or args.v045_candidate_proof is not None:
            parser.error("formal warm-start cannot use shortlist-only evidence")
    return args


def main() -> int:
    args = _parse_args()
    evidence = validate_source_evidence(
        args.source_checkpoint,
        args.v045_training_summary,
        args.v045_candidate_proof,
        engineering_smoke=args.engineering_smoke,
        source_approval_path=args.v045_source_approval,
    )
    layouts = inspect_live_observation_contract()
    checkpoint_path, proof_path = _next_output_paths(
        args.output_dir.resolve(), args.engineering_smoke
    )
    result = transform_checkpoint(
        args.source_checkpoint.resolve(),
        args.target_template.resolve(),
        checkpoint_path,
        evidence,
        layouts,
    )
    payload = {
        "schema_version": "v049_v045_shape_aware_warmstart_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **evidence,
        **result,
        "source_code_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in (
                REPO_ROOT
                / "experiments/v045_rcss_t1_getup_adaptation/rcss_mjlab_task.py",
                EXPERIMENT_ROOT / "rcss_mjlab_task.py",
                EXPERIMENT_ROOT / "mdp.py",
                EXPERIMENT_ROOT / "scripts/prepare_v049_warmstart.py",
                EXPERIMENT_ROOT / "scripts/train_v049.py",
                EXPERIMENT_ROOT / "configs/v049_competitive_acceptance.json",
            )
        },
        "transform": {
            "actor": "source [0:75] -> target [0:75]; target [75:81] zero",
            "critic": "source [0:75] -> target [0:75]; source [75:78] -> target [81:84]; all other new columns zero",
            "compatible_hidden_output_bias_and_distribution_tensors_copied": True,
            "optimizer_state_reset": True,
            "iteration_reset": True,
            "curriculum_counter_reset": True,
        },
        "formal_training_allowed": evidence["formal_source_approved"],
        "training_completed": False,
        "fixed_validation_passed": False,
        "heldout_validation_passed": False,
        "quality_passed": False,
    }
    if proof_path.exists():
        raise FileExistsError(proof_path)
    proof_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "provenance": str(proof_path),
                "formal_training_allowed": payload["formal_training_allowed"],
                "quality_passed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
