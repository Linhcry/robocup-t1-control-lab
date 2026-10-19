"""Fail-closed v045 -> v046 network-shape warm start.

The two policies share the first 75 proprioceptive actor inputs, the same
512-256-128 MLP and the same 23 actions.  v046 appends nine football inputs.
Its privileged critic additionally moves the v045 three-dimensional base
linear velocity from columns 75:78 to columns 84:87.  This module performs only
that explicit mapping; it never guesses how an unknown checkpoint is laid out.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch


TRANSFORM_VERSION = "v046_v045_shape_warmstart_v1"
V045_ACTOR_DIM = 75
V046_ACTOR_DIM = 84
V045_CRITIC_DIM = 78
V046_CRITIC_DIM = 101
SHARED_PROPRIO_SLICE = (0, 75)
V045_BASE_LIN_SLICE = (75, 78)
V046_BASE_LIN_SLICE = (84, 87)

_ACTOR_TAIL_SHAPES = {
    "distribution.log_std_param": (23,),
    "mlp.0.bias": (512,),
    "mlp.2.weight": (256, 512),
    "mlp.2.bias": (256,),
    "mlp.4.weight": (128, 256),
    "mlp.4.bias": (128,),
    "mlp.6.weight": (23, 128),
    "mlp.6.bias": (23,),
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value: torch.Tensor) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise TypeError("checkpoint state-dict values must be tensors")
    return tuple(value.shape)


def _require_shape(
    state: dict[str, torch.Tensor], key: str, expected: tuple[int, ...]
) -> None:
    if key not in state:
        raise ValueError(f"checkpoint is missing {key!r}")
    actual = _shape(state[key])
    if actual != expected:
        raise ValueError(f"{key} has shape {actual}, expected {expected}")


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
        raise ValueError(
            "v045 checkpoint top-level keys differ from the audited contract: "
            f"{sorted(payload)}"
        )
    actor = payload["actor_state_dict"]
    critic = payload["critic_state_dict"]
    if not isinstance(actor, dict) or not isinstance(critic, dict):
        raise ValueError("actor/critic state dictionaries are invalid")
    expected_actor = {"mlp.0.weight", *_ACTOR_TAIL_SHAPES}
    expected_critic = {"mlp.0.weight", *_CRITIC_TAIL_SHAPES, *_NORMALIZER_KEYS}
    if set(actor) != expected_actor:
        raise ValueError(f"unexpected v045 actor keys: {sorted(set(actor) ^ expected_actor)}")
    if set(critic) != expected_critic:
        raise ValueError(
            f"unexpected v045 critic keys: {sorted(set(critic) ^ expected_critic)}"
        )
    _require_shape(actor, "mlp.0.weight", (512, V045_ACTOR_DIM))
    for key, shape in _ACTOR_TAIL_SHAPES.items():
        _require_shape(actor, key, shape)
    _require_shape(critic, "mlp.0.weight", (512, V045_CRITIC_DIM))
    for key, shape in _CRITIC_TAIL_SHAPES.items():
        _require_shape(critic, key, shape)
    for key in _NORMALIZER_KEYS[:3]:
        _require_shape(critic, key, (1, V045_CRITIC_DIM))
    _require_shape(critic, "obs_normalizer.count", ())

    optimizer = payload["optimizer_state_dict"]
    if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
        raise ValueError("v045 optimizer state has an unknown structure")
    groups = optimizer["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("v045 optimizer must contain exactly one parameter group")
    if groups[0].get("params") != list(range(17)):
        raise ValueError("v045 optimizer parameter order is not the audited 17-parameter order")
    return actor, critic


def transform_payload(
    source_payload: dict[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    learning_rate: float = 5.0e-4,
) -> dict[str, Any]:
    """Return a new v046-shaped checkpoint without mutating the source."""

    actor, critic = _validate_source_payload(source_payload)
    if not (learning_rate > 0.0):
        raise ValueError("learning rate must be positive")

    target_actor: dict[str, torch.Tensor] = {
        "obs_normalizer._mean": torch.zeros(
            (1, V046_ACTOR_DIM), dtype=actor["mlp.0.weight"].dtype
        ),
        "obs_normalizer._var": torch.ones(
            (1, V046_ACTOR_DIM), dtype=actor["mlp.0.weight"].dtype
        ),
        "obs_normalizer._std": torch.ones(
            (1, V046_ACTOR_DIM), dtype=actor["mlp.0.weight"].dtype
        ),
        "obs_normalizer.count": torch.zeros((), dtype=torch.int64),
    }
    actor_first = torch.zeros(
        (512, V046_ACTOR_DIM), dtype=actor["mlp.0.weight"].dtype
    )
    actor_first[:, :V045_ACTOR_DIM].copy_(actor["mlp.0.weight"])
    target_actor["mlp.0.weight"] = actor_first
    for key in _ACTOR_TAIL_SHAPES:
        target_actor[key] = actor[key].detach().clone()

    target_critic: dict[str, torch.Tensor] = {}
    for statistic, fill in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", 1.0),
        ("obs_normalizer._std", 1.0),
    ):
        value = torch.full(
            (1, V046_CRITIC_DIM),
            fill,
            dtype=critic[statistic].dtype,
        )
        value[:, :V045_ACTOR_DIM].copy_(critic[statistic][:, :V045_ACTOR_DIM])
        value[:, V046_BASE_LIN_SLICE[0] : V046_BASE_LIN_SLICE[1]].copy_(
            critic[statistic][:, V045_BASE_LIN_SLICE[0] : V045_BASE_LIN_SLICE[1]]
        )
        target_critic[statistic] = value
    target_critic["obs_normalizer.count"] = critic[
        "obs_normalizer.count"
    ].detach().clone()
    critic_first = torch.zeros(
        (512, V046_CRITIC_DIM), dtype=critic["mlp.0.weight"].dtype
    )
    critic_first[:, :V045_ACTOR_DIM].copy_(critic["mlp.0.weight"][:, :V045_ACTOR_DIM])
    critic_first[:, V046_BASE_LIN_SLICE[0] : V046_BASE_LIN_SLICE[1]].copy_(
        critic["mlp.0.weight"][:, V045_BASE_LIN_SLICE[0] : V045_BASE_LIN_SLICE[1]]
    )
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
                "transform": TRANSFORM_VERSION,
                "source": str(source_path.resolve()),
                "source_sha256": source_sha256,
                "source_iteration": int(source_payload["iter"]),
                "actor_shared_columns": [0, 75],
                "actor_new_zero_columns": [75, 84],
                "critic_shared_columns": [0, 75],
                "critic_base_lin_source_columns": [75, 78],
                "critic_base_lin_target_columns": [84, 87],
                "critic_new_zero_columns": [[75, 84], [87, 101]],
                "optimizer_state_reset": True,
                "iteration_reset": True,
                "curriculum_counter_reset": True,
                "learning_rate_reset_to": float(learning_rate),
            },
        },
    }


def transform_checkpoint(
    source: Path, target: Path, *, learning_rate: float = 5.0e-4
) -> dict[str, Any]:
    """Create one immutable transformed checkpoint and return its provenance."""

    source = source.resolve()
    target = target.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite warm-start artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    transformed = transform_payload(
        payload,
        source_path=source,
        source_sha256=source_sha,
        learning_rate=learning_rate,
    )
    torch.save(transformed, target)
    return {
        "transform": TRANSFORM_VERSION,
        "source": {"path": str(source), "sha256": source_sha},
        "target": {"path": str(target), "sha256": sha256_file(target)},
        "learning_rate": float(learning_rate),
    }


def audit_mapping(source: Path, target: Path) -> dict[str, bool]:
    """Verify every copied and zero-initialized tensor independently."""

    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    source_actor, source_critic = _validate_source_payload(source_payload)
    target_payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(target_payload, dict):
        raise ValueError("target checkpoint payload is invalid")
    target_actor = target_payload.get("actor_state_dict", {})
    target_critic = target_payload.get("critic_state_dict", {})

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
        "target_top_level_keys_exact": set(target_payload)
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

    checks.update(
        {
            "actor_first_layer_shape_512x84": _shape(target_actor["mlp.0.weight"])
            == (512, 84),
            "actor_shared_75_columns_exact": torch.equal(
                target_actor["mlp.0.weight"][:, :75],
                source_actor["mlp.0.weight"],
            ),
            "actor_new_9_columns_zero": bool(
                torch.count_nonzero(target_actor["mlp.0.weight"][:, 75:84]) == 0
            ),
            "actor_new_normalizer_is_identity": bool(
                torch.count_nonzero(target_actor["obs_normalizer._mean"]) == 0
                and torch.all(target_actor["obs_normalizer._var"] == 1)
                and torch.all(target_actor["obs_normalizer._std"] == 1)
                and int(target_actor["obs_normalizer.count"].item()) == 0
            ),
            "actor_tail_and_distribution_exact": all(
                torch.equal(target_actor[key], source_actor[key])
                for key in _ACTOR_TAIL_SHAPES
            ),
            "critic_first_layer_shape_512x101": _shape(
                target_critic["mlp.0.weight"]
            )
            == (512, 101),
            "critic_shared_75_columns_exact": torch.equal(
                target_critic["mlp.0.weight"][:, :75],
                source_critic["mlp.0.weight"][:, :75],
            ),
            "critic_base_lin_75_78_mapped_to_84_87": torch.equal(
                target_critic["mlp.0.weight"][:, 84:87],
                source_critic["mlp.0.weight"][:, 75:78],
            ),
            "critic_all_new_columns_zero": bool(
                torch.count_nonzero(target_critic["mlp.0.weight"][:, 75:84]) == 0
                and torch.count_nonzero(target_critic["mlp.0.weight"][:, 87:101])
                == 0
            ),
            "critic_normalizer_shared_and_base_lin_mapped": all(
                torch.equal(target_critic[key][:, :75], source_critic[key][:, :75])
                and torch.equal(
                    target_critic[key][:, 84:87], source_critic[key][:, 75:78]
                )
                for key in _NORMALIZER_KEYS[:3]
            ),
            "critic_new_normalizer_columns_identity": all(
                bool(
                    torch.count_nonzero(target_critic[key][:, 75:84]) == 0
                    and torch.count_nonzero(target_critic[key][:, 87:101]) == 0
                )
                if key.endswith("_mean")
                else bool(
                    torch.all(target_critic[key][:, 75:84] == 1)
                    and torch.all(target_critic[key][:, 87:101] == 1)
                )
                for key in _NORMALIZER_KEYS[:3]
            ),
            "critic_normalizer_count_copied": torch.equal(
                target_critic["obs_normalizer.count"],
                source_critic["obs_normalizer.count"],
            ),
            "critic_tail_exact": all(
                torch.equal(target_critic[key], source_critic[key])
                for key in _CRITIC_TAIL_SHAPES
            ),
            "optimizer_state_reset": target_payload.get(
                "optimizer_state_dict", {}
            ).get("state")
            == {},
            "iteration_reset_to_zero": target_payload.get("iter") == 0,
            "curriculum_counter_reset_to_zero": target_payload.get("infos", {})
            .get("env_state", {})
            .get("common_step_counter")
            == 0,
            "provenance_declares_current_transform": target_payload.get(
                "infos", {}
            )
            .get("warm_start_provenance", {})
            .get("transform")
            == TRANSFORM_VERSION,
            "all_target_tensors_finite": all(
                torch.isfinite(value).all().item()
                for state in (target_actor, target_critic)
                for value in state.values()
                if value.is_floating_point()
            ),
        }
    )
    return {name: bool(value) for name, value in checks.items()}
