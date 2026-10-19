#!/usr/bin/env python3
"""Audit a v045->v048 warm-start and one completed CPU PPO update."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v048_fullbody_legal_tackle.scripts._evidence import (  # noqa: E402
    write_new_json,
)
from experiments.v048_fullbody_legal_tackle.warmstart import (  # noqa: E402
    audit_mapping,
    inspect_live_observation_contract,
    sha256,
    transform_source_manifest,
    validate_source_evidence,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--v045-source-approval", type=Path)
    parser.add_argument("--v045-training-summary", type=Path)
    parser.add_argument("--v045-candidate-proof", type=Path)
    parser.add_argument("--warmstart-checkpoint", type=Path, required=True)
    parser.add_argument("--warmstart-provenance", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    if args.engineering_smoke:
        if args.v045_source_approval is not None:
            parser.error("engineering audit cannot consume a formal approval manifest")
        if args.v045_training_summary is None or args.v045_candidate_proof is None:
            parser.error("engineering audit requires summary and candidate proof")
    else:
        if args.v045_source_approval is None:
            parser.error("formal audit requires --v045-source-approval")
        if args.v045_training_summary is not None or args.v045_candidate_proof is not None:
            parser.error("formal audit accepts only the unified approval manifest")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _yaml_scalar(path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.*?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip("'\"")
    raise ValueError(f"missing YAML key {key}: {path}")


def _optimizer_has_completed_update(payload: dict[str, Any]) -> bool:
    optimizer = payload.get("optimizer_state_dict", {})
    state = optimizer.get("state") if isinstance(optimizer, dict) else None
    if not isinstance(state, dict) or not state:
        return False
    steps: list[float] = []
    for value in state.values():
        if isinstance(value, dict) and "step" in value:
            step = value["step"]
            steps.append(float(step.item()) if isinstance(step, torch.Tensor) else float(step))
    return bool(steps) and min(steps) >= 1.0


def main() -> int:
    args = _args()
    source = args.source_checkpoint.resolve()
    summary = (
        args.v045_training_summary.resolve()
        if args.v045_training_summary is not None
        else None
    )
    candidate = (
        args.v045_candidate_proof.resolve()
        if args.v045_candidate_proof is not None
        else None
    )
    approval = (
        args.v045_source_approval.resolve()
        if args.v045_source_approval is not None
        else None
    )
    warmstart = args.warmstart_checkpoint.resolve()
    provenance_path = args.warmstart_provenance.resolve()
    training_run = args.training_run.resolve()
    for path in (
        source,
        *(item for item in (summary, candidate, approval) if item is not None),
        warmstart,
        provenance_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not training_run.is_dir():
        raise FileNotFoundError(training_run)

    source_binding = validate_source_evidence(
        source,
        summary,
        candidate,
        approval_manifest_path=approval,
        engineering_smoke=args.engineering_smoke,
    )
    live_layout = inspect_live_observation_contract()
    mapping_checks = audit_mapping(source, warmstart)
    provenance = _read_json(provenance_path)
    current_manifest = transform_source_manifest()
    output_checkpoints = sorted(training_run.glob("model_*.pt"))
    if len(output_checkpoints) != 1:
        raise ValueError(
            f"one-iteration smoke must contain exactly one model checkpoint, found "
            f"{len(output_checkpoints)}"
        )
    output_checkpoint = output_checkpoints[0]
    warmstart_payload = torch.load(warmstart, map_location="cpu", weights_only=False)
    output_payload = torch.load(
        output_checkpoint, map_location="cpu", weights_only=False
    )
    agent_yaml = training_run / "params" / "agent.yaml"
    env_yaml = training_run / "params" / "env.yaml"
    event_files = sorted(training_run.glob("events.out.tfevents.*"))
    if not agent_yaml.is_file() or not env_yaml.is_file():
        raise FileNotFoundError("training run is missing agent/env YAML")

    actor_changed = any(
        not torch.equal(
            warmstart_payload["actor_state_dict"][key],
            output_payload["actor_state_dict"][key],
        )
        for key in (
            "mlp.0.weight",
            "mlp.0.bias",
            "mlp.2.weight",
            "mlp.4.weight",
            "mlp.6.weight",
            "mlp.6.bias",
        )
    )
    critic_changed = any(
        not torch.equal(
            warmstart_payload["critic_state_dict"][key],
            output_payload["critic_state_dict"][key],
        )
        for key in (
            "mlp.0.weight",
            "mlp.0.bias",
            "mlp.2.weight",
            "mlp.4.weight",
            "mlp.6.weight",
            "mlp.6.bias",
        )
    )
    checks = {
        **mapping_checks,
        "live_v045_actor_layout_is_75": live_layout["v045_actor"][-1][1][1]
        == 75,
        "live_v045_critic_layout_is_78": live_layout["v045_critic"][-1][1][1]
        == 78,
        "live_v048_actor_layout_is_93": live_layout["v048_actor"][-1][1][1]
        == 93,
        "live_v048_critic_layout_is_190": live_layout["v048_critic"][-1][1][1]
        == 190,
        "provenance_schema_current": provenance.get("schema_version")
        == "v048_v045_semantic_warmstart_provenance_v1",
        "provenance_binds_exact_source_chain": provenance.get("source_binding")
        == source_binding,
        "provenance_binds_cached_checkpoint": provenance.get(
            "cached_checkpoint", {}
        ).get("path")
        == str(warmstart)
        and provenance.get("cached_checkpoint", {}).get("sha256")
        == sha256(warmstart),
        "provenance_binds_current_transform_source_hashes": bool(
            isinstance(provenance.get("transform_source_manifest"), dict)
            and set(provenance["transform_source_manifest"])
            == set(current_manifest)
            and all(
                provenance["transform_source_manifest"][name].get("sha256")
                == current_manifest[name]["sha256"]
                for name in current_manifest
            )
        ),
        "formal_mode_flag_matches_request": provenance.get(
            "formal_source_binding"
        )
        is (not args.engineering_smoke),
        "warmstart_optimizer_empty": warmstart_payload[
            "optimizer_state_dict"
        ]["state"]
        == {},
        "warmstart_iteration_zero": warmstart_payload.get("iter") == 0,
        "warmstart_curriculum_zero": warmstart_payload.get("infos", {})
        .get("env_state", {})
        .get("common_step_counter")
        == 0,
        "training_output_has_target_actor_shape": output_payload[
            "actor_state_dict"
        ]["mlp.0.weight"].shape
        == (512, 93),
        "training_output_has_target_critic_shape": output_payload[
            "critic_state_dict"
        ]["mlp.0.weight"].shape
        == (512, 190),
        "training_output_optimizer_completed_update": _optimizer_has_completed_update(
            output_payload
        ),
        "training_output_actor_weights_updated": actor_changed,
        "training_output_critic_weights_updated": critic_changed,
        "training_output_advanced_environment_counter": output_payload.get(
            "infos", {}
        )
        .get("env_state", {})
        .get("common_step_counter", 0)
        >= 24,
        "agent_yaml_resumes_from_cached_run": provenance_path.parent.name
        in _yaml_scalar(agent_yaml, "load_run"),
        "agent_yaml_loads_cached_checkpoint": warmstart.name
        in _yaml_scalar(agent_yaml, "load_checkpoint"),
        "one_tensorboard_event_file_exists": len(event_files) == 1,
        "source_checkpoint_unchanged": sha256(source)
        == source_binding["source_checkpoint"]["sha256"],
    }
    passed = bool(checks) and all(bool(value) for value in checks.values())
    payload = {
        "schema_version": "v048_warmstart_semantic_audit_v1",
        "source_binding": source_binding,
        "source_checkpoint": {
            "path": str(source),
            "sha256": sha256(source),
        },
        "warmstart_checkpoint": {
            "path": str(warmstart),
            "sha256": sha256(warmstart),
        },
        "warmstart_provenance": {
            "path": str(provenance_path),
            "sha256": sha256(provenance_path),
        },
        "training_run": str(training_run),
        "training_output_checkpoint": {
            "path": str(output_checkpoint.resolve()),
            "sha256": sha256(output_checkpoint),
        },
        "live_observation_contract": live_layout,
        "checks": {name: bool(value) for name, value in checks.items()},
        "warmstart_semantic_audit_passed": passed,
        "engineering_smoke": bool(args.engineering_smoke),
        "training_completed": False,
        "fixed_validation_passed": False,
        "heldout_validation_passed": False,
        "quality_passed": False,
        "conclusion": (
            "This proves semantic column mapping, strict evidence binding, CPU "
            "checkpoint loading and one PPO update only. It is not tackle ability."
        ),
    }
    output = write_new_json("v048_warmstart_semantic_audit", payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "warmstart_semantic_audit_passed": passed,
                "quality_passed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
