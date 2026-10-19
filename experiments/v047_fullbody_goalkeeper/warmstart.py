"""Auditable v045-to-v047 shape-aware warm-start transformation.

The transfer is deliberately narrow: the 75 deployable proprioceptive inputs,
the v045 critic's three privileged base-linear-velocity inputs, all compatible
hidden/output layers, and the action distribution are inherited.  Every new
goalkeeper input column starts at exactly zero so no untrained ball feature can
silently perturb the source motor prior.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.v045_rcss_t1_getup_adaptation.source_approval import (
    validate_source_approval,
)


SOURCE_ACTOR_DIM = 75
SOURCE_CRITIC_DIM = 78
TARGET_ACTOR_DIM = 87
TARGET_CRITIC_DIM = 107
ACTION_DIM = 23
SHARED_PROPRIO_SLICE = (0, 75)
SOURCE_BASE_LIN_SLICE = (75, 78)
TARGET_BASE_LIN_SLICE = (87, 90)
FORMAL_SHORTLIST_SEEDS = [45100, 45101, 45102]
TRANSFORM_VERSION = "v047_v045_shape_aware_warmstart_v2"
EXPERIMENT_ROOT = Path(__file__).resolve().parent

SOURCE_ACTOR_SHAPES = {
    "distribution.log_std_param": (ACTION_DIM,),
    "mlp.0.weight": (512, SOURCE_ACTOR_DIM),
    "mlp.0.bias": (512,),
    "mlp.2.weight": (256, 512),
    "mlp.2.bias": (256,),
    "mlp.4.weight": (128, 256),
    "mlp.4.bias": (128,),
    "mlp.6.weight": (ACTION_DIM, 128),
    "mlp.6.bias": (ACTION_DIM,),
}
SOURCE_CRITIC_SHAPES = {
    "obs_normalizer._mean": (1, SOURCE_CRITIC_DIM),
    "obs_normalizer._var": (1, SOURCE_CRITIC_DIM),
    "obs_normalizer._std": (1, SOURCE_CRITIC_DIM),
    "obs_normalizer.count": (),
    "mlp.0.weight": (512, SOURCE_CRITIC_DIM),
    "mlp.0.bias": (512,),
    "mlp.2.weight": (256, 512),
    "mlp.2.bias": (256,),
    "mlp.4.weight": (128, 256),
    "mlp.4.bias": (128,),
    "mlp.6.weight": (1, 128),
    "mlp.6.bias": (1,),
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 for a local evidence or checkpoint file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_source_manifest() -> dict[str, str]:
    """Hash every local source that determines staging or target semantics."""

    paths = {
        "warmstart.py": EXPERIMENT_ROOT / "warmstart.py",
        "rcss_goalkeeper_task.py": EXPERIMENT_ROOT / "rcss_goalkeeper_task.py",
        "scripts/train_v047.py": EXPERIMENT_ROOT / "scripts" / "train_v047.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _checkpoint_entry(summary: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    entries = summary.get("checkpoints")
    if not isinstance(entries, list):
        raise ValueError("v045 training summary has no checkpoint list")
    matching = [
        item
        for item in entries
        if isinstance(item, dict) and item.get("name") == checkpoint.name
    ]
    if len(matching) != 1:
        raise ValueError("source checkpoint is not uniquely listed in v045 summary")
    return matching[0]


def validate_formal_source(
    checkpoint: Path,
    source_approval_path: Path,
) -> dict[str, Any]:
    """Require the terminal fixed+held-out v045 approval for formal transfer."""

    binding = validate_source_approval(checkpoint, source_approval_path)
    if binding.get("formal_source_approved") is not True:
        raise ValueError("v045 downstream source approval did not authorize transfer")
    return binding


def validate_engineering_source(
    checkpoint: Path, engineering_proof_path: Path
) -> dict[str, Any]:
    """Bind an explicitly non-promotable checkpoint for CPU plumbing tests."""

    checkpoint = checkpoint.resolve()
    engineering_proof_path = engineering_proof_path.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    proof = _read_json(engineering_proof_path, "engineering source proof")
    if proof.get("schema_version") != "v047_v045_engineering_source_v1":
        raise ValueError("engineering source proof schema mismatch")
    source = proof.get("source_checkpoint")
    if not isinstance(source, dict):
        raise ValueError("engineering source proof has no source checkpoint")
    checkpoint_hash = sha256(checkpoint)
    if Path(str(source.get("path", ""))).resolve() != checkpoint:
        raise ValueError("engineering proof refers to a different checkpoint")
    if source.get("sha256") != checkpoint_hash:
        raise ValueError("engineering source checkpoint hash mismatch")
    if proof.get("engineering_smoke_only") is not True:
        raise ValueError("engineering proof is not smoke-only")
    if proof.get("promotable") is not False or proof.get("quality_passed") is not False:
        raise ValueError("engineering proof must explicitly be non-promotable")
    return {
        "mode": "engineering_smoke",
        "source_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
        },
        "engineering_source_proof": {
            "path": str(engineering_proof_path),
            "sha256": sha256(engineering_proof_path),
        },
        "promotable_source_binding": False,
    }


def _require_state_shapes(
    state: dict[str, Any], expected: dict[str, tuple[int, ...]], label: str
) -> None:
    import torch

    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            f"{label} keys changed; missing={missing}, unexpected={unexpected}"
        )
    for key, shape in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            actual = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value)
            raise ValueError(f"{label}.{key} shape {actual} != {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{label}.{key} contains non-finite values")


def transform_checkpoint(
    source_checkpoint: Path,
    source_binding: dict[str, Any],
    *,
    learning_rate: float,
) -> dict[str, Any]:
    """Return a target-shaped, iteration-zero v047 checkpoint payload."""

    import torch

    source_checkpoint = source_checkpoint.resolve()
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    required = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"v045 checkpoint is missing fields: {missing}")
    actor = payload["actor_state_dict"]
    critic = payload["critic_state_dict"]
    if not isinstance(actor, dict) or not isinstance(critic, dict):
        raise ValueError("v045 actor/critic state dictionaries are invalid")
    _require_state_shapes(actor, SOURCE_ACTOR_SHAPES, "v045 actor")
    _require_state_shapes(critic, SOURCE_CRITIC_SHAPES, "v045 critic")

    target_actor = {key: value.detach().clone() for key, value in actor.items()}
    actor_first = actor["mlp.0.weight"]
    expanded_actor_first = actor_first.new_zeros((512, TARGET_ACTOR_DIM))
    expanded_actor_first[:, :SOURCE_ACTOR_DIM] = actor_first
    target_actor["mlp.0.weight"] = expanded_actor_first

    target_critic = {key: value.detach().clone() for key, value in critic.items()}
    critic_first = critic["mlp.0.weight"]
    expanded_critic_first = critic_first.new_zeros((512, TARGET_CRITIC_DIM))
    expanded_critic_first[:, :SOURCE_ACTOR_DIM] = critic_first[:, :SOURCE_ACTOR_DIM]
    expanded_critic_first[:, TARGET_BASE_LIN_SLICE[0] : TARGET_BASE_LIN_SLICE[1]] = (
        critic_first[:, SOURCE_BASE_LIN_SLICE[0] : SOURCE_BASE_LIN_SLICE[1]]
    )
    target_critic["mlp.0.weight"] = expanded_critic_first

    # Preserve v045 running moments for the 75 common values and move the three
    # base-linear-velocity moments to v047 indices 87:90.  New privileged fields
    # start with neutral moments; their first-layer weights are zero.
    for key, neutral in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", 1.0),
        ("obs_normalizer._std", 1.0),
    ):
        source_moments = critic[key]
        target_moments = source_moments.new_full((1, TARGET_CRITIC_DIM), neutral)
        target_moments[:, :SOURCE_ACTOR_DIM] = source_moments[:, :SOURCE_ACTOR_DIM]
        target_moments[:, TARGET_BASE_LIN_SLICE[0] : TARGET_BASE_LIN_SLICE[1]] = (
            source_moments[
                :, SOURCE_BASE_LIN_SLICE[0] : SOURCE_BASE_LIN_SLICE[1]
            ]
        )
        target_critic[key] = target_moments

    optimizer = copy.deepcopy(payload["optimizer_state_dict"])
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("param_groups"), list):
        raise ValueError("v045 optimizer state is invalid")
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("v045 optimizer must contain exactly one parameter group")
    if optimizer["param_groups"][0].get("params") != list(range(17)):
        raise ValueError("v045 optimizer parameter ordering changed")
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = float(learning_rate)

    source_iteration = int(payload["iter"])
    mapping = {
        "actor": {
            "source_input_dim": SOURCE_ACTOR_DIM,
            "target_input_dim": TARGET_ACTOR_DIM,
            "copied_columns": [0, 75],
            "zero_initialized_columns": [75, 87],
        },
        "critic": {
            "source_input_dim": SOURCE_CRITIC_DIM,
            "target_input_dim": TARGET_CRITIC_DIM,
            "copied_proprio_columns": {"source": [0, 75], "target": [0, 75]},
            "copied_base_lin_vel_columns": {
                "source": list(SOURCE_BASE_LIN_SLICE),
                "target": list(TARGET_BASE_LIN_SLICE),
            },
            "zero_initialized_target_columns": [[75, 87], [90, 107]],
        },
        "compatible_later_layers_copied": ["mlp.0.bias", "mlp.2", "mlp.4", "mlp.6"],
        "action_distribution_copied": "distribution.log_std_param",
        "critic_running_moments_remapped": True,
    }
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
                "source_iteration": source_iteration,
                "mapping": mapping,
                "optimizer_state_reset": True,
                "iteration_reset": True,
                "curriculum_counter_reset": True,
                "learning_rate_reset_to": float(learning_rate),
                "quality_passed": False,
            },
        },
    }


def prepare_cached_warmstart(
    source_checkpoint: Path,
    source_binding: dict[str, Any],
    log_root: Path,
    *,
    learning_rate: float,
) -> tuple[str, str, Path]:
    """Create or verify one content-addressed checkpoint and provenance JSON."""

    import torch

    source_checkpoint = source_checkpoint.resolve()
    log_root = log_root.resolve()
    binding_json = json.dumps(source_binding, sort_keys=True, separators=(",", ":"))
    binding_hash = hashlib.sha256(binding_json.encode("utf-8")).hexdigest()
    source_hash = sha256(source_checkpoint)
    source_manifest = transform_source_manifest()
    transform_hash = hashlib.sha256(
        json.dumps(source_manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    mode_tag = "formal" if source_binding["mode"] == "formal" else "engineering"
    run_name = (
        f"bootstrap_v045_{source_hash[:16]}_{binding_hash[:12]}_"
        f"{transform_hash[:12]}_{mode_tag}_shape_warmstart_v2"
    )
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / f"{source_checkpoint.stem}.v047_warmstart.pt"
    proof_path = run_dir / "warmstart_provenance.json"

    if not target.exists():
        transformed = transform_checkpoint(
            source_checkpoint, source_binding, learning_rate=learning_rate
        )
        torch.save(transformed, target)
    target_hash = sha256(target)
    provenance = {
        "schema_version": "v047_v045_shape_aware_warmstart_provenance_v2",
        "source_binding": source_binding,
        "source_iteration": int(
            torch.load(source_checkpoint, map_location="cpu", weights_only=False)["iter"]
        ),
        "cached_checkpoint": {"path": str(target), "sha256": target_hash},
        "transform": TRANSFORM_VERSION,
        "transform_source_manifest": source_manifest,
        "mapping": {
            "actor_shared": {"source": [0, 75], "target": [0, 75]},
            "actor_new_zero": [75, 87],
            "critic_shared": {"source": [0, 75], "target": [0, 75]},
            "critic_base_lin_vel": {"source": [75, 78], "target": [87, 90]},
            "critic_new_zero": [[75, 87], [90, 107]],
        },
        "later_layers_copied": True,
        "action_distribution_copied": True,
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
    if proof_path.exists():
        existing = json.loads(proof_path.read_text(encoding="utf-8"))
        if existing != provenance:
            raise RuntimeError(f"existing warm-start provenance mismatch: {proof_path}")
    else:
        proof_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return run_name, target.name, proof_path
