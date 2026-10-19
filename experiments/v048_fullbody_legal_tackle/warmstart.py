"""Fail-closed, semantic v045 -> v048 full-body policy warm-start.

v045 and v048 use the same 23 RCSS joints and 512-256-128 MLP, but their
observation columns are not in the same order.  This module copies only the
explicitly audited semantic slices, zero-initializes every new football column,
and resets all training state.  It never treats a non-formal v045 probe as a
formal source unless the caller explicitly selects engineering-smoke mode.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parent
SOURCE_ACTOR_DIM = 75
SOURCE_CRITIC_DIM = 78
TARGET_ACTOR_DIM = 93
TARGET_CRITIC_DIM = 190
ACTION_DIM = 23
TARGET_LEARNING_RATE = 3.0e-4
FORMAL_SHORTLIST_SEEDS = [45100, 45101, 45102]
TRANSFORM_VERSION = "v048_v045_semantic_shape_warmstart_v1"

SOURCE_ACTOR_LAYOUT = (
    ("base_ang_vel", (0, 3)),
    ("projected_gravity", (3, 6)),
    ("joint_pos", (6, 29)),
    ("joint_vel", (29, 52)),
    ("actions", (52, 75)),
)
SOURCE_CRITIC_LAYOUT = SOURCE_ACTOR_LAYOUT + (("base_lin_vel", (75, 78)),)
TARGET_SHARED_LAYOUT = {
    "projected_gravity": (0, 3),
    "base_ang_vel": (3, 6),
    "base_lin_vel": (6, 9),
    "joint_pos": (9, 32),
    "joint_vel": (32, 55),
    "actions": (55, 78),
}

ACTOR_MAPPING = (
    ("base_ang_vel", (0, 3), (3, 6)),
    ("projected_gravity", (3, 6), (0, 3)),
    ("joint_pos", (6, 29), (9, 32)),
    ("joint_vel", (29, 52), (32, 55)),
    ("actions", (52, 75), (55, 78)),
)
CRITIC_MAPPING = ACTOR_MAPPING + (("base_lin_vel", (75, 78), (6, 9)),)

_ACTOR_TAIL_SHAPES = {
    "distribution.log_std_param": (ACTION_DIM,),
    "mlp.0.bias": (512,),
    "mlp.2.weight": (256, 512),
    "mlp.2.bias": (256,),
    "mlp.4.weight": (128, 256),
    "mlp.4.bias": (128,),
    "mlp.6.weight": (ACTION_DIM, 128),
    "mlp.6.bias": (ACTION_DIM,),
}
_CRITIC_TAIL_SHAPES = {
    "mlp.0.bias": (512,),
    "mlp.2.weight": (256, 512),
    "mlp.2.bias": (256,),
    "mlp.4.weight": (128, 256),
    "mlp.4.bias": (128,),
    "mlp.6.weight": (1, 128),
    "mlp.6.bias": (1,),
}
_NORMALIZER_KEYS = (
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
    "obs_normalizer.count",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _checkpoint_ref_matches(
    reference: Any, checkpoint: Path, checkpoint_hash: str
) -> bool:
    return bool(
        isinstance(reference, dict)
        and isinstance(reference.get("path"), str)
        and Path(reference["path"]).expanduser().resolve() == checkpoint
        and reference.get("sha256") == checkpoint_hash
    )


def _checkpoint_entry(
    summary: dict[str, Any], checkpoint: Path, checkpoint_hash: str
) -> dict[str, Any]:
    matches = [
        row
        for row in summary.get("checkpoints", [])
        if isinstance(row, dict)
        and Path(str(row.get("path", ""))).expanduser().resolve() == checkpoint
        and row.get("sha256") == checkpoint_hash
    ]
    if len(matches) != 1:
        raise ValueError(
            "v045 checkpoint is not uniquely path/hash bound in its training summary"
        )
    return matches[0]


def validate_source_evidence(
    checkpoint: Path,
    training_summary_path: Path | None,
    candidate_proof_path: Path | None,
    *,
    approval_manifest_path: Path | None = None,
    engineering_smoke: bool,
) -> dict[str, Any]:
    """Return the one permitted formal or engineering source binding.

    Formal downstream use has exactly one authority: the shared v045 approval
    manifest, which revalidates audit -> training -> candidate -> fixed ->
    held-out evidence for the same checkpoint.  Direct summary/candidate
    binding remains available only for an explicitly non-promotable engineering
    smoke so shape transformation and one CPU PPO update can be tested before a
    formal v045 source exists.
    """

    checkpoint = checkpoint.expanduser().resolve()
    if not engineering_smoke:
        if approval_manifest_path is None:
            raise ValueError(
                "formal v048 warm-start requires --v045-source-approval; "
                "summary/candidate evidence is not a formal authorization"
            )
        if training_summary_path is not None or candidate_proof_path is not None:
            raise ValueError(
                "formal v048 warm-start accepts only the unified v045 approval "
                "manifest, not direct summary/candidate arguments"
            )
        from experiments.v045_rcss_t1_getup_adaptation.source_approval import (
            validate_source_approval,
        )

        binding = validate_source_approval(
            checkpoint, approval_manifest_path.expanduser().resolve()
        )
        if (
            binding.get("mode") != "formal"
            or binding.get("formal_source_approved") is not True
            or binding.get("promotable_source_binding") is not True
        ):
            raise ValueError("unified v045 approval did not authorize this source")
        return binding

    if approval_manifest_path is not None:
        raise ValueError(
            "engineering smoke must use its explicitly non-formal "
            "summary/candidate chain, not a formal approval manifest"
        )
    if training_summary_path is None or candidate_proof_path is None:
        raise ValueError(
            "engineering warm-start requires both v045 training summary and "
            "candidate proof"
        )
    training_summary_path = training_summary_path.expanduser().resolve()
    candidate_proof_path = candidate_proof_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    summary = _read_json(training_summary_path, "v045 training summary")
    candidate = _read_json(candidate_proof_path, "v045 candidate proof")
    summary_hash = sha256(training_summary_path)
    candidate_hash = sha256(candidate_proof_path)

    if summary.get("schema_version") != "v045_mjlab_training_summary_v1":
        raise ValueError("v045 training summary schema mismatch")
    if summary.get("run_complete") is not True:
        raise ValueError("v045 training summary is incomplete")
    run_directory = Path(str(summary.get("run_directory", ""))).resolve()
    if checkpoint.parent != run_directory:
        raise ValueError("v045 checkpoint is outside its summarized run")
    _checkpoint_entry(summary, checkpoint, checkpoint_hash)

    if candidate.get("schema_version") != "v045_target_candidate_eval_v1":
        raise ValueError("v045 candidate proof schema mismatch")
    proof_summary = candidate.get("training_summary")
    if not isinstance(proof_summary, dict):
        raise ValueError("v045 candidate proof does not bind a training summary")
    if Path(str(proof_summary.get("path", ""))).resolve() != training_summary_path:
        raise ValueError("v045 candidate proof binds a different training summary")
    if proof_summary.get("sha256") != summary_hash:
        raise ValueError("v045 candidate proof training-summary hash mismatch")
    if Path(str(candidate.get("run_directory", ""))).resolve() != run_directory:
        raise ValueError("v045 candidate proof refers to a different training run")
    matching_results = [
        row
        for row in candidate.get("results", [])
        if isinstance(row, dict)
        and _checkpoint_ref_matches(
            row.get("checkpoint"), checkpoint, checkpoint_hash
        )
    ]
    if len(matching_results) != 1:
        raise ValueError("v045 candidate proof does not uniquely evaluate the checkpoint")
    if candidate.get("candidate_shortlist_only") is not True:
        raise ValueError("v045 candidate proof is not shortlist-only evidence")
    if candidate.get("quality_passed") is not False:
        raise ValueError("v045 candidate proof must not claim final quality")

    if summary.get("formal_training_run") is not False:
        raise ValueError("engineering mode requires an explicitly non-formal summary")
    if candidate.get("formal_target_shortlist_protocol") is not False:
        raise ValueError("engineering mode requires a non-formal candidate proof")
    if candidate.get("selected_checkpoint_for_rcss_fixed_validation") is not None:
        raise ValueError("engineering proof must not select a formal checkpoint")
    mode = "engineering_smoke"

    return {
        "mode": mode,
        "source_checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash},
        "v045_training_summary": {
            "path": str(training_summary_path),
            "sha256": summary_hash,
            "schema_version": summary["schema_version"],
        },
        "v045_candidate_proof": {
            "path": str(candidate_proof_path),
            "sha256": candidate_hash,
            "schema_version": candidate["schema_version"],
        },
        "formal_source_approved": False,
        "promotable_source_binding": False,
    }


def _live_manager_layout(env: Any, group: str) -> list[tuple[str, tuple[int, int]]]:
    manager = env.observation_manager
    offset = 0
    result: list[tuple[str, tuple[int, int]]] = []
    for name, shape in zip(
        manager.active_terms[group], manager.group_obs_term_dim[group], strict=True
    ):
        size = 1
        for value in shape:
            size *= int(value)
        result.append((name, (offset, offset + size)))
        offset += size
    return result


def inspect_live_observation_contract() -> dict[str, Any]:
    """Construct the real CPU tasks and prove every source/target input slice."""

    from mjlab.envs import ManagerBasedRlEnv
    from experiments.v045_rcss_t1_getup_adaptation.rcss_mjlab_task import (
        rcss_t1_getup_env_cfg,
    )
    from experiments.v048_fullbody_legal_tackle.rcss_mjlab_task import (
        V048_ACTOR_OBSERVATION_LAYOUT,
        defender_local_observation_components,
        exact_rcss_tackle_env_cfg,
    )

    source_cfg = rcss_t1_getup_env_cfg()
    source_cfg.scene.num_envs = 1
    source_cfg.scene.terrain.num_envs = 1
    source_env = ManagerBasedRlEnv(cfg=source_cfg, device="cpu")
    try:
        source_actor = _live_manager_layout(source_env, "actor")
        source_critic = _live_manager_layout(source_env, "critic")
    finally:
        source_env.close()
    if tuple(source_actor) != SOURCE_ACTOR_LAYOUT:
        raise ValueError(
            f"live v045 actor layout drifted: expected={SOURCE_ACTOR_LAYOUT}, "
            f"actual={source_actor}"
        )
    if tuple(source_critic) != SOURCE_CRITIC_LAYOUT:
        raise ValueError(
            f"live v045 critic layout drifted: expected={SOURCE_CRITIC_LAYOUT}, "
            f"actual={source_critic}"
        )

    target_cfg = exact_rcss_tackle_env_cfg()
    target_cfg.scene.num_envs = 1
    target_env = ManagerBasedRlEnv(cfg=target_cfg, device="cpu")
    try:
        observations, _ = target_env.reset(seed=48000)
        components = defender_local_observation_components(target_env)
        target_actor = [
            (name, tuple(bounds)) for name, bounds in V048_ACTOR_OBSERVATION_LAYOUT
        ]
        live_component_widths = {
            name: int(value.shape[-1]) for name, value in components.items()
        }
        target_critic_terms = _live_manager_layout(target_env, "critic")
        actor_shape = tuple(observations["actor"].shape)
        critic_shape = tuple(observations["critic"].shape)
    finally:
        target_env.close()
    if list(components) != [name for name, _ in target_actor]:
        raise ValueError("live v048 component order differs from its task contract")
    for name, (start, stop) in target_actor:
        if live_component_widths[name] != stop - start:
            raise ValueError(f"live v048 component width drifted for {name}")
    if actor_shape[-1] != TARGET_ACTOR_DIM or critic_shape[-1] != TARGET_CRITIC_DIM:
        raise ValueError(
            f"live target shapes drifted: actor={actor_shape}, critic={critic_shape}"
        )
    if target_critic_terms != [
        ("local_deployable_state", (0, 93)),
        ("privileged_world_state", (93, 190)),
    ]:
        raise ValueError(f"live v048 critic term layout drifted: {target_critic_terms}")
    return {
        "v045_actor": [[name, list(bounds)] for name, bounds in source_actor],
        "v045_critic": [[name, list(bounds)] for name, bounds in source_critic],
        "v048_actor": [[name, list(bounds)] for name, bounds in target_actor],
        "v048_critic": [
            [name, list(bounds)] for name, bounds in target_critic_terms
        ],
        "v048_live_component_widths": live_component_widths,
        "actor_mapping": [
            {"semantic": name, "source": list(source), "target": list(target)}
            for name, source, target in ACTOR_MAPPING
        ],
        "critic_mapping": [
            {"semantic": name, "source": list(source), "target": list(target)}
            for name, source, target in CRITIC_MAPPING
        ],
    }


def _shape(value: torch.Tensor) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise TypeError("checkpoint state values must be tensors")
    return tuple(value.shape)


def _validate_source_payload(payload: Any) -> tuple[dict, dict]:
    if not isinstance(payload, dict):
        raise ValueError("v045 checkpoint payload is not a dictionary")
    required_top = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
    }
    if set(payload) != required_top:
        raise ValueError(f"v045 checkpoint top-level keys drifted: {sorted(payload)}")
    actor = payload["actor_state_dict"]
    critic = payload["critic_state_dict"]
    expected_actor = {"mlp.0.weight", *_ACTOR_TAIL_SHAPES}
    expected_critic = {
        "mlp.0.weight",
        *_CRITIC_TAIL_SHAPES,
        *_NORMALIZER_KEYS,
    }
    if set(actor) != expected_actor:
        raise ValueError(f"unexpected v045 actor keys: {sorted(actor)}")
    if set(critic) != expected_critic:
        raise ValueError(f"unexpected v045 critic keys: {sorted(critic)}")
    if _shape(actor["mlp.0.weight"]) != (512, SOURCE_ACTOR_DIM):
        raise ValueError("v045 actor first layer is not 512x75")
    if _shape(critic["mlp.0.weight"]) != (512, SOURCE_CRITIC_DIM):
        raise ValueError("v045 critic first layer is not 512x78")
    for key, expected in _ACTOR_TAIL_SHAPES.items():
        if _shape(actor[key]) != expected:
            raise ValueError(f"v045 actor {key} shape mismatch")
    for key, expected in _CRITIC_TAIL_SHAPES.items():
        if _shape(critic[key]) != expected:
            raise ValueError(f"v045 critic {key} shape mismatch")
    for key in _NORMALIZER_KEYS[:3]:
        if _shape(critic[key]) != (1, SOURCE_CRITIC_DIM):
            raise ValueError(f"v045 critic {key} shape mismatch")
    if _shape(critic["obs_normalizer.count"]) != ():
        raise ValueError("v045 critic normalizer count shape mismatch")
    optimizer = payload["optimizer_state_dict"]
    if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
        raise ValueError("v045 optimizer structure changed")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("v045 optimizer must have one parameter group")
    if groups[0].get("params") != list(range(17)):
        raise ValueError("v045 optimizer parameter ordering changed")
    return actor, critic


def _copy_mapped_columns(
    source: torch.Tensor,
    target: torch.Tensor,
    mapping: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...],
) -> None:
    target.zero_()
    for _, (source_start, source_stop), (target_start, target_stop) in mapping:
        if source_stop - source_start != target_stop - target_start:
            raise ValueError("warm-start mapping width mismatch")
        target[..., target_start:target_stop].copy_(
            source[..., source_start:source_stop]
        )


def transform_payload(
    source_payload: dict[str, Any],
    source_binding: dict[str, Any],
    *,
    learning_rate: float = TARGET_LEARNING_RATE,
) -> dict[str, Any]:
    """Return a new v048-shaped checkpoint without changing the v045 source."""

    actor, critic = _validate_source_payload(source_payload)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    target_actor: dict[str, torch.Tensor] = {
        "obs_normalizer._mean": torch.zeros(
            (1, TARGET_ACTOR_DIM), dtype=actor["mlp.0.weight"].dtype
        ),
        "obs_normalizer._var": torch.ones(
            (1, TARGET_ACTOR_DIM), dtype=actor["mlp.0.weight"].dtype
        ),
        "obs_normalizer._std": torch.ones(
            (1, TARGET_ACTOR_DIM), dtype=actor["mlp.0.weight"].dtype
        ),
        "obs_normalizer.count": torch.zeros((), dtype=torch.int64),
    }
    actor_first = torch.zeros(
        (512, TARGET_ACTOR_DIM), dtype=actor["mlp.0.weight"].dtype
    )
    _copy_mapped_columns(actor["mlp.0.weight"], actor_first, ACTOR_MAPPING)
    target_actor["mlp.0.weight"] = actor_first
    for key in _ACTOR_TAIL_SHAPES:
        target_actor[key] = actor[key].detach().clone()

    target_critic: dict[str, torch.Tensor] = {}
    for key, neutral in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", 1.0),
        ("obs_normalizer._std", 1.0),
    ):
        value = torch.full(
            (1, TARGET_CRITIC_DIM), neutral, dtype=critic[key].dtype
        )
        for _, (source_start, source_stop), (target_start, target_stop) in CRITIC_MAPPING:
            value[:, target_start:target_stop].copy_(
                critic[key][:, source_start:source_stop]
            )
        target_critic[key] = value
    target_critic["obs_normalizer.count"] = critic[
        "obs_normalizer.count"
    ].detach().clone()
    critic_first = torch.zeros(
        (512, TARGET_CRITIC_DIM), dtype=critic["mlp.0.weight"].dtype
    )
    _copy_mapped_columns(critic["mlp.0.weight"], critic_first, CRITIC_MAPPING)
    target_critic["mlp.0.weight"] = critic_first
    for key in _CRITIC_TAIL_SHAPES:
        target_critic[key] = critic[key].detach().clone()

    optimizer = copy.deepcopy(source_payload["optimizer_state_dict"])
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = float(learning_rate)

    return {
        "actor_state_dict": target_actor,
        "critic_state_dict": target_critic,
        "optimizer_state_dict": optimizer,
        "iter": 0,
        "infos": {
            "env_state": {"common_step_counter": 0},
            "warm_start_provenance": {
                "schema_version": TRANSFORM_VERSION,
                "source_binding": source_binding,
                "source_iteration": int(source_payload["iter"]),
                "actor_mapping": [
                    {
                        "semantic": name,
                        "source": list(source),
                        "target": list(target),
                    }
                    for name, source, target in ACTOR_MAPPING
                ],
                "actor_new_zero_columns": [78, 93],
                "critic_mapping": [
                    {
                        "semantic": name,
                        "source": list(source),
                        "target": list(target),
                    }
                    for name, source, target in CRITIC_MAPPING
                ],
                "critic_new_zero_columns": [78, 190],
                "compatible_later_layers_copied": True,
                "action_distribution_copied": "distribution.log_std_param",
                "optimizer_state_reset": True,
                "iteration_reset": True,
                "curriculum_counter_reset": True,
                "learning_rate_reset_to": float(learning_rate),
                "quality_passed": False,
            },
        },
    }


def _mapped_target_columns(
    mapping: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]
) -> list[int]:
    result: list[int] = []
    for _, _, (start, stop) in mapping:
        result.extend(range(start, stop))
    return result


def audit_mapping(source_path: Path, target_path: Path) -> dict[str, bool]:
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    target = torch.load(target_path, map_location="cpu", weights_only=False)
    source_actor, source_critic = _validate_source_payload(source)
    target_actor = target.get("actor_state_dict", {})
    target_critic = target.get("critic_state_dict", {})
    actor_expected = {
        "mlp.0.weight",
        *_ACTOR_TAIL_SHAPES,
        *_NORMALIZER_KEYS,
    }
    critic_expected = {
        "mlp.0.weight",
        *_CRITIC_TAIL_SHAPES,
        *_NORMALIZER_KEYS,
    }
    checks: dict[str, bool] = {
        "target_top_level_keys_exact": set(target)
        == {
            "actor_state_dict",
            "critic_state_dict",
            "optimizer_state_dict",
            "iter",
            "infos",
        },
        "target_actor_keys_exact": set(target_actor) == actor_expected,
        "target_critic_keys_exact": set(target_critic) == critic_expected,
    }
    if not checks["target_actor_keys_exact"] or not checks["target_critic_keys_exact"]:
        return checks
    actor_mapped = _mapped_target_columns(ACTOR_MAPPING)
    critic_mapped = _mapped_target_columns(CRITIC_MAPPING)
    actor_new = [index for index in range(TARGET_ACTOR_DIM) if index not in actor_mapped]
    critic_new = [
        index for index in range(TARGET_CRITIC_DIM) if index not in critic_mapped
    ]
    actor_columns_exact = all(
        torch.equal(
            target_actor["mlp.0.weight"][:, target_start:target_stop],
            source_actor["mlp.0.weight"][:, source_start:source_stop],
        )
        for _, (source_start, source_stop), (target_start, target_stop) in ACTOR_MAPPING
    )
    critic_columns_exact = all(
        torch.equal(
            target_critic["mlp.0.weight"][:, target_start:target_stop],
            source_critic["mlp.0.weight"][:, source_start:source_stop],
        )
        for _, (source_start, source_stop), (target_start, target_stop) in CRITIC_MAPPING
    )
    normalizers_mapped = all(
        torch.equal(
            target_critic[key][:, target_start:target_stop],
            source_critic[key][:, source_start:source_stop],
        )
        for key in _NORMALIZER_KEYS[:3]
        for _, (source_start, source_stop), (target_start, target_stop) in CRITIC_MAPPING
    )
    checks.update(
        {
            "actor_first_layer_shape_512x93": _shape(
                target_actor["mlp.0.weight"]
            )
            == (512, TARGET_ACTOR_DIM),
            "actor_semantic_columns_exact": actor_columns_exact,
            "actor_football_opponent_columns_zero": bool(
                torch.count_nonzero(target_actor["mlp.0.weight"][:, actor_new])
                == 0
            ),
            "actor_normalizer_identity": bool(
                torch.count_nonzero(target_actor["obs_normalizer._mean"]) == 0
                and torch.all(target_actor["obs_normalizer._var"] == 1)
                and torch.all(target_actor["obs_normalizer._std"] == 1)
                and int(target_actor["obs_normalizer.count"].item()) == 0
            ),
            "actor_later_layers_and_distribution_exact": all(
                torch.equal(target_actor[key], source_actor[key])
                for key in _ACTOR_TAIL_SHAPES
            ),
            "critic_first_layer_shape_512x190": _shape(
                target_critic["mlp.0.weight"]
            )
            == (512, TARGET_CRITIC_DIM),
            "critic_semantic_columns_exact": critic_columns_exact,
            "critic_all_new_columns_zero": bool(
                torch.count_nonzero(target_critic["mlp.0.weight"][:, critic_new])
                == 0
            ),
            "critic_normalizer_semantic_columns_exact": normalizers_mapped,
            "critic_new_normalizer_columns_identity": bool(
                torch.count_nonzero(
                    target_critic["obs_normalizer._mean"][:, critic_new]
                )
                == 0
                and torch.all(
                    target_critic["obs_normalizer._var"][:, critic_new] == 1
                )
                and torch.all(
                    target_critic["obs_normalizer._std"][:, critic_new] == 1
                )
            ),
            "critic_normalizer_count_copied": torch.equal(
                target_critic["obs_normalizer.count"],
                source_critic["obs_normalizer.count"],
            ),
            "critic_later_layers_exact": all(
                torch.equal(target_critic[key], source_critic[key])
                for key in _CRITIC_TAIL_SHAPES
            ),
            "optimizer_state_reset": target.get("optimizer_state_dict", {}).get(
                "state"
            )
            == {},
            "iteration_reset_to_zero": target.get("iter") == 0,
            "curriculum_counter_reset_to_zero": target.get("infos", {})
            .get("env_state", {})
            .get("common_step_counter")
            == 0,
            "transform_version_bound": target.get("infos", {})
            .get("warm_start_provenance", {})
            .get("schema_version")
            == TRANSFORM_VERSION,
            "all_target_tensors_finite": all(
                bool(torch.isfinite(value).all())
                for state in (target_actor, target_critic)
                for value in state.values()
                if value.is_floating_point()
            ),
        }
    )
    return {name: bool(value) for name, value in checks.items()}


def transform_source_manifest() -> dict[str, dict[str, str]]:
    paths = {
        "warmstart": EXPERIMENT_ROOT / "warmstart.py",
        "task": EXPERIMENT_ROOT / "rcss_mjlab_task.py",
        "safe_actions": EXPERIMENT_ROOT / "safe_actions.py",
        "training_entry": EXPERIMENT_ROOT / "scripts" / "train_v048.py",
    }
    return {
        name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in paths.items()
    }


def prepare_cached_warmstart(
    source_checkpoint: Path,
    source_binding: dict[str, Any],
    log_root: Path,
    *,
    learning_rate: float = TARGET_LEARNING_RATE,
) -> tuple[str, str, Path]:
    """Create or verify one content-addressed immutable warm-start artifact."""

    source_checkpoint = source_checkpoint.resolve()
    log_root = log_root.resolve()
    live_observation_contract = inspect_live_observation_contract()
    source_hash = sha256(source_checkpoint)
    binding_hash = hashlib.sha256(
        json.dumps(source_binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_manifest = transform_source_manifest()
    transform_hash = hashlib.sha256(
        json.dumps(source_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    mode_tag = "formal" if source_binding["mode"] == "formal" else "engineering"
    run_name = (
        f"bootstrap_v045_{source_hash[:16]}_{binding_hash[:12]}_"
        f"{transform_hash[:12]}_{mode_tag}_semantic_warmstart_v1"
    )
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / f"{source_checkpoint.stem}.v048_warmstart.pt"
    provenance_path = run_dir / "warmstart_provenance.json"
    if not target.exists():
        source_payload = torch.load(
            source_checkpoint, map_location="cpu", weights_only=False
        )
        transformed = transform_payload(
            source_payload, source_binding, learning_rate=learning_rate
        )
        torch.save(transformed, target)
    mapping_checks = audit_mapping(source_checkpoint, target)
    if not mapping_checks or not all(mapping_checks.values()):
        failed = [name for name, value in mapping_checks.items() if not value]
        raise RuntimeError(f"warm-start mapping audit failed: {failed}")
    provenance = {
        "schema_version": "v048_v045_semantic_warmstart_provenance_v1",
        "source_binding": source_binding,
        "source_iteration": int(
            torch.load(source_checkpoint, map_location="cpu", weights_only=False)[
                "iter"
            ]
        ),
        "cached_checkpoint": {"path": str(target), "sha256": sha256(target)},
        "transform": TRANSFORM_VERSION,
        "transform_source_manifest": source_manifest,
        "live_observation_contract": live_observation_contract,
        "mapping_checks": mapping_checks,
        "optimizer_state_reset": True,
        "iteration_reset_to": 0,
        "curriculum_counter_reset_to": 0,
        "learning_rate_reset_to": float(learning_rate),
        "formal_source_binding": source_binding["mode"] == "formal",
        "training_completed": False,
        "fixed_validation_completed": False,
        "held_out_test_completed": False,
        "quality_passed": False,
    }
    if provenance_path.exists():
        existing = _read_json(provenance_path, "existing warm-start provenance")
        if existing != provenance:
            raise RuntimeError(
                f"existing content-addressed provenance differs: {provenance_path}"
            )
    else:
        provenance_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return run_name, target.name, provenance_path
