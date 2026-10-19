"""Seal completion and provenance evidence for one v045 MJLab run.

This file proves that a run finished with the intended target model, reset
contract, source checkpoint and saved checkpoint set.  It deliberately cannot
promote behavior; the independent RCSS fixed and held-out evaluators do that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")
TARGET_AUDIT_SCHEMA = "v045_target_model_semantic_audit_v10"
TRAINING_SUMMARY_SCHEMA = "v045_mjlab_training_summary_v2"
WATCHDOG_REPORT_SCHEMA = "v045_stability_report_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml_scalar(path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.*?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip("'\"")
    raise ValueError(f"could not find top-level YAML field {key!r} in {path}")


def _integer_from_yaml(path: Path, key: str) -> int:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return int(match.group(1))
    raise ValueError(f"could not find integer {key!r} in {path}")


def _scalar_summary(events: list[Any]) -> dict[str, Any]:
    values = np.asarray([float(event.value) for event in events], dtype=np.float64)
    steps = np.asarray([int(event.step) for event in events], dtype=np.int64)
    maximum = int(np.argmax(values))
    minimum = int(np.argmin(values))
    return {
        "count": len(events),
        "first": {"step": int(steps[0]), "value": float(values[0])},
        "last": {"step": int(steps[-1]), "value": float(values[-1])},
        "minimum": {"step": int(steps[minimum]), "value": float(values[minimum])},
        "maximum": {"step": int(steps[maximum]), "value": float(values[maximum])},
        "last_50_mean": float(np.mean(values[-min(50, len(values)) :])),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _live_reference(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _require_live_reference(reference: Any, label: str) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} reference is missing")
    path = Path(reference.get("path", "")).expanduser().resolve()
    if not path.is_file() or reference.get("sha256") != _sha256(path):
        raise ValueError(f"{label} file/hash changed: {path}")
    return path


def _validate_source_chain(
    selection_path: Path, provenance_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = _load_json(selection_path)
    if selection.get("schema_version") != "v044_source_candidate_eval_v2":
        raise ValueError("source selection proof schema mismatch")
    if not selection.get("formal_source_shortlist_protocol", False):
        raise ValueError("source selection was not the formal protocol")
    if selection.get("seeds") != [44700, 44701, 44702]:
        raise ValueError("source selection seed set mismatch")
    if selection.get("num_environments") != 256 or selection.get("steps") != 299:
        raise ValueError("source selection sample size mismatch")
    selected = selection.get("selected_checkpoint_for_target_adaptation")
    if not isinstance(selected, dict) or not selected.get("sha256"):
        raise ValueError("source selection has no selected checkpoint")

    provenance = _load_json(provenance_path)
    required_true = (
        "optimizer_state_reset",
        "curriculum_counter_reset",
    )
    if provenance.get("transform") != "network_warm_start_with_fresh_training_state_v2":
        raise ValueError("bootstrap transform mismatch")
    if not all(provenance.get(key) is True for key in required_true):
        raise ValueError("bootstrap training-state reset proof is incomplete")
    if provenance.get("target_iteration") != 0:
        raise ValueError("bootstrap target iteration was not reset")
    if provenance.get("source_sha256") != selected["sha256"]:
        raise ValueError("bootstrap source differs from the formal selected source")
    source = Path(provenance.get("source", "")).resolve()
    cached = Path(provenance.get("cached_checkpoint", "")).resolve()
    if not source.is_file() or _sha256(source) != provenance.get("source_sha256"):
        raise ValueError("bootstrap source file/hash mismatch")
    if not cached.is_file() or _sha256(cached) != provenance.get("cached_sha256"):
        raise ValueError("cached warm-start file/hash mismatch")
    selection_ref = provenance.get("selection_proof")
    if not isinstance(selection_ref, dict):
        raise ValueError("bootstrap provenance does not bind source selection")
    if selection_ref.get("sha256") != _sha256(selection_path):
        raise ValueError("bootstrap provenance selection hash mismatch")
    if selection_ref.get("selected_checkpoint_sha256") != selected["sha256"]:
        raise ValueError("bootstrap provenance selected hash mismatch")
    return selection, provenance


def _validate_target_audit(path: Path) -> dict[str, Any]:
    audit = _load_json(path)
    if audit.get("schema_version") != TARGET_AUDIT_SCHEMA:
        raise ValueError("v045 target audit schema mismatch")
    checks = audit.get("checks")
    if (
        not audit.get("audit_completed", False)
        or not audit.get("target_model_semantics_passed", False)
        or not isinstance(checks, dict)
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise ValueError("v045 target audit is incomplete or contains a failure")
    for source in audit.get("sources", {}).values():
        source_path = Path(source.get("path", "")).resolve()
        if not source_path.is_file() or _sha256(source_path) != source.get("sha256"):
            raise ValueError(f"target audit source file/hash changed: {source_path}")
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-proof", type=Path, required=True)
    parser.add_argument("--bootstrap-provenance", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--expected-iterations", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_iterations <= 0:
        parser.error("expected-iterations must be positive")
    return args


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    selection_path = args.selection_proof.resolve()
    provenance_path = args.bootstrap_provenance.resolve()
    audit_path = args.target_audit.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical evidence: {output}")
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    selection, provenance = _validate_source_chain(selection_path, provenance_path)
    target_audit = _validate_target_audit(audit_path)
    selected_source_sha = selection["selected_checkpoint_for_target_adaptation"][
        "sha256"
    ]
    if (
        target_audit.get("source_action_semantics", {})
        .get("checkpoint", {})
        .get("sha256")
        != selected_source_sha
    ):
        raise ValueError(
            "target audit measured a different checkpoint than the selected source"
        )

    agent_yaml = run_dir / "params" / "agent.yaml"
    env_yaml = run_dir / "params" / "env.yaml"
    if not agent_yaml.is_file() or not env_yaml.is_file():
        raise FileNotFoundError("run is missing params/agent.yaml or params/env.yaml")
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise ValueError(f"expected one TensorBoard event file, found {len(event_files)}")

    run_contract_path = run_dir / "run_contract.json"
    training_profile_path = run_dir / "training_profile.json"
    watchdog_report_path = run_dir / "stability" / "stability_report.json"
    watchdog_config_path = run_dir / "stability" / "watchdog_config.json"
    for required in (
        run_contract_path,
        training_profile_path,
        watchdog_report_path,
        watchdog_config_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"run is missing required safety evidence: {required}")
    run_contract = _load_json(run_contract_path)
    training_profile = _load_json(training_profile_path)
    watchdog_report = _load_json(watchdog_report_path)
    watchdog_config = _load_json(watchdog_config_path)
    if run_contract.get("schema_version") != "v045_training_run_contract_v1":
        raise ValueError("v045 training run-contract schema mismatch")
    if run_contract.get("training_profile") != training_profile:
        raise ValueError("run contract and training-profile evidence differ")
    _require_live_reference(run_contract.get("entrypoint"), "v045 training entrypoint")
    if watchdog_report.get("schema_version") != WATCHDOG_REPORT_SCHEMA:
        raise ValueError("v045 stability watchdog report schema mismatch")
    if watchdog_config.get("schema_version") != "v045_physics_substep_watchdog_v2":
        raise ValueError("v045 stability watchdog config schema mismatch")
    if watchdog_report.get("training_profile") != training_profile:
        raise ValueError("watchdog report and training profile differ")
    if watchdog_config.get("training_profile") != training_profile:
        raise ValueError("watchdog config and training profile differ")
    _require_live_reference(
        watchdog_report.get("watchdog_config"), "v045 watchdog config"
    )
    _require_live_reference(
        watchdog_config.get("watchdog_source"), "v045 watchdog source"
    )
    counts = watchdog_report.get("counts")
    required_zero_counts = (
        "warning_environment_substeps",
        "abort_environment_substeps",
        "nonfinite_environment_substeps",
        "joint_limit_violation_environment_substeps",
        "constraint_capacity_near_miss_environment_substeps",
        "persistent_event_files",
    )
    watchdog_clean = bool(
        isinstance(counts, dict)
        and all(counts.get(name) == 0 for name in required_zero_counts)
        and watchdog_report.get("status") == "completed"
        and watchdog_report.get("completed_cleanly_with_zero_events") is True
        and watchdog_report.get("per_physics_substep_hook") is True
        and int(watchdog_report.get("physics_substeps_checked", 0)) > 0
    )
    if not watchdog_clean:
        raise ValueError("v045 training did not complete with a clean substep watchdog")

    checkpoints: list[dict[str, Any]] = []
    for checkpoint_path in run_dir.glob("model_*.pt"):
        match = CHECKPOINT_RE.match(checkpoint_path.name)
        if match:
            checkpoints.append(
                {
                    "iteration": int(match.group(1)),
                    "name": checkpoint_path.name,
                    "path": str(checkpoint_path.resolve()),
                    "size_bytes": checkpoint_path.stat().st_size,
                    "sha256": _sha256(checkpoint_path),
                }
            )
    checkpoints.sort(key=lambda item: item["iteration"])
    if not checkpoints:
        raise ValueError("run has no model_N.pt checkpoints")

    accumulator = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    scalars = {
        tag: _scalar_summary(accumulator.Scalars(tag))
        for tag in scalar_tags
        if accumulator.Scalars(tag)
    }
    metric_tag = "Episode_Metrics/getup_success"
    metric_events = accumulator.Scalars(metric_tag)
    if not metric_events:
        raise ValueError(f"run is missing required scalar {metric_tag}")
    metric_by_step = {int(event.step): float(event.value) for event in metric_events}
    for checkpoint in checkpoints:
        checkpoint["training_getup_success_at_iteration"] = metric_by_step.get(
            checkpoint["iteration"]
        )

    num_envs = _integer_from_yaml(env_yaml, "num_envs")
    env_seed = _integer_from_yaml(env_yaml, "seed")
    agent_seed = _integer_from_yaml(agent_yaml, "seed")
    steps_per_env = _integer_from_yaml(agent_yaml, "num_steps_per_env")
    configured_iterations = _integer_from_yaml(agent_yaml, "max_iterations")
    completed_iterations = max(int(event.step) for event in metric_events) + 1
    run_complete = (
        completed_iterations >= args.expected_iterations
        and configured_iterations == args.expected_iterations
        and checkpoints[-1]["iteration"] >= args.expected_iterations - 1
    )
    cached_checkpoint = Path(provenance["cached_checkpoint"]).resolve()
    load_checkpoint = _yaml_scalar(agent_yaml, "load_checkpoint")
    resume_value = _yaml_scalar(agent_yaml, "resume").lower()
    bootstrap_bound_to_run = (
        resume_value == "true"
        and cached_checkpoint.name in load_checkpoint
        and provenance_path.parent.name in _yaml_scalar(agent_yaml, "load_run")
    )
    formal_training_run = bool(
        run_complete
        and args.expected_iterations == 3000
        and configured_iterations == 3000
        and completed_iterations >= 3000
        and num_envs == 4096
        and steps_per_env == 24
        and env_seed == 45000
        and agent_seed == 45000
        and bootstrap_bound_to_run
        and target_audit["target_model_semantics_passed"]
        and run_contract.get("formal_training_run") is True
        and run_contract.get("engineering_smoke") is False
        and run_contract.get("num_environments") == 4096
        and run_contract.get("iterations") == 3000
        and run_contract.get("environment_seed") == 45000
        and run_contract.get("agent_seed") == 45000
        and run_contract.get("nan_guard_enabled") is True
        and isinstance(run_contract.get("profile_selection_proof"), dict)
        and training_profile.get("selected", {}).get("solver") == "rcss_mjb"
        and training_profile.get("selected", {}).get("reset") == "rcss_eval_v2"
        and training_profile.get("formal_components_match") is True
        and watchdog_clean
    )
    profile_selection_path = None
    if isinstance(run_contract.get("profile_selection_proof"), dict):
        profile_selection_path = _require_live_reference(
            run_contract["profile_selection_proof"],
            "v045 profile-selection proof",
        )

    payload = {
        "schema_version": TRAINING_SUMMARY_SCHEMA,
        "experiment": "v045_rcss_t1_getup_adaptation",
        "run_directory": str(run_dir),
        "expected_iterations": args.expected_iterations,
        "configured_iterations": configured_iterations,
        "completed_iterations_from_events": completed_iterations,
        "run_complete": run_complete,
        "formal_training_run": formal_training_run,
        "num_environments": num_envs,
        "steps_per_environment_per_iteration": steps_per_env,
        "environment_seed": env_seed,
        "agent_seed": agent_seed,
        "total_environment_steps": num_envs * steps_per_env * completed_iterations,
        "agent_config": {"path": str(agent_yaml), "sha256": _sha256(agent_yaml)},
        "environment_config": {"path": str(env_yaml), "sha256": _sha256(env_yaml)},
        "tensorboard_event": {
            "path": str(event_files[0]),
            "sha256": _sha256(event_files[0]),
        },
        "source_selection_proof": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
            "selected_checkpoint": selection[
                "selected_checkpoint_for_target_adaptation"
            ],
        },
        "bootstrap_provenance": {
            "path": str(provenance_path),
            "sha256": _sha256(provenance_path),
            "source_sha256": provenance["source_sha256"],
            "cached_sha256": provenance["cached_sha256"],
            "bootstrap_bound_to_run": bootstrap_bound_to_run,
        },
        "target_model_audit": {
            "path": str(audit_path),
            "sha256": _sha256(audit_path),
            "schema_version": target_audit["schema_version"],
            "target_model_semantics_passed": True,
        },
        "run_contract": _live_reference(run_contract_path),
        "training_profile": {
            **_live_reference(training_profile_path),
            "selected": training_profile.get("selected"),
            "formal_components_match": training_profile.get(
                "formal_components_match"
            ),
        },
        "profile_selection_proof": (
            None
            if profile_selection_path is None
            else _live_reference(profile_selection_path)
        ),
        "stability_watchdog": {
            "report": _live_reference(watchdog_report_path),
            "config": _live_reference(watchdog_config_path),
            "true_physics_substep_hook": True,
            "completed_cleanly_with_zero_events": watchdog_clean,
            "physics_substeps_checked": watchdog_report.get(
                "physics_substeps_checked"
            ),
            "counts": counts,
        },
        "checkpoints": checkpoints,
        "scalar_summaries": scalars,
        "training_metric_is_not_promotion_evidence": True,
        "validation_passed": False,
        "quality_passed": False,
        "reason_quality_is_false": (
            "A completed, provenance-bound PPO run is only candidate generation. "
            "The selected checkpoint must still pass exact-RCSS four-pose fixed "
            "validation and the disjoint held-out test, including the locomotion "
            "handoff and every safety gate."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "run_complete": run_complete,
                "formal_training_run": formal_training_run,
                "completed_iterations": completed_iterations,
                "checkpoints": len(checkpoints),
                "quality_passed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if run_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
