#!/usr/bin/env python3
"""Independent audit of v049 shape mapping, fail-closed proofs and CPU update."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v049_fullbody_receive.scripts.prepare_v049_warmstart import (  # noqa: E402
    sha256,
    validate_source_evidence,
)
from experiments.v049_fullbody_receive.scripts.train_v049 import (  # noqa: E402
    _validate_warm_start,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--ppo-proof", type=Path, required=True)
    parser.add_argument("--v045-source-approval", type=Path)
    return parser.parse_args()


def _next_output() -> Path:
    root = EXPERIMENT_ROOT / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    for run in range(1, 10_000):
        candidate = root / f"v049_warmstart_semantic_audit_run_{run:02d}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused v049 warm-start audit filename")


def _all_true(value: object) -> bool:
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def main() -> int:
    args = _parse_args()
    artifact = args.artifact.resolve()
    provenance_path = args.provenance.resolve()
    ppo_path = args.ppo_proof.resolve()
    for path in (artifact, provenance_path, ppo_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    ppo = json.loads(ppo_path.read_text(encoding="utf-8"))
    formal_source = provenance.get("formal_source_approved") is True
    if formal_source and args.v045_source_approval is None:
        raise ValueError("formal warm-start audit requires --v045-source-approval")
    if not formal_source and args.v045_source_approval is not None:
        raise ValueError("engineering warm-start audit cannot consume a formal approval")
    source_ref = provenance["source_checkpoint"]
    source_path = Path(source_ref["path"]).resolve()
    summary_ref = provenance.get("v045_training_summary")
    candidate_ref = provenance.get("v045_candidate_proof")
    summary_path = (
        Path(summary_ref["path"]).resolve() if isinstance(summary_ref, dict) else None
    )
    candidate_path = (
        Path(candidate_ref["path"]).resolve() if isinstance(candidate_ref, dict) else None
    )
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    target = torch.load(artifact, map_location="cpu", weights_only=False)

    source_evidence_valid = False
    source_evidence_error = ""
    try:
        validate_source_evidence(
            source_path,
            summary_path,
            candidate_path,
            engineering_smoke=not formal_source,
            source_approval_path=args.v045_source_approval,
        )
    except (ValueError, FileNotFoundError) as exc:
        source_evidence_error = str(exc)
    else:
        source_evidence_valid = True
    training_loader_valid = False
    training_loader_error = ""
    try:
        _validate_warm_start(
            artifact,
            provenance_path,
            not formal_source,
            args.v045_source_approval,
        )
    except (ValueError, FileNotFoundError) as exc:
        training_loader_error = str(exc)
    else:
        training_loader_valid = True

    actor_source = source["actor_state_dict"]
    critic_source = source["critic_state_dict"]
    actor_target = target["actor_state_dict"]
    critic_target = target["critic_state_dict"]
    recorded_sources = provenance.get("source_code_sha256", {})
    source_hashes_current = bool(recorded_sources) and all(
        (REPO_ROOT / relative).is_file()
        and sha256(REPO_ROOT / relative) == expected
        for relative, expected in recorded_sources.items()
    )
    checks = {
        "artifact_hash_matches_provenance": provenance.get("output_checkpoint", {}).get("sha256")
        == sha256(artifact),
        "source_checkpoint_hash_matches": source_ref.get("sha256") == sha256(source_path),
        "source_evidence_mode_revalidated": source_evidence_valid,
        "training_loader_mode_revalidated": training_loader_valid,
        "all_recorded_source_hashes_are_current": source_hashes_current,
        "live_layout_and_mapping_checks_passed": _all_true(provenance.get("mapping_checks")),
        "actor_source_shape_is_512x75": tuple(actor_source["mlp.0.weight"].shape)
        == (512, 75),
        "actor_target_shape_is_512x81": tuple(actor_target["mlp.0.weight"].shape)
        == (512, 81),
        "actor_first_75_exact": torch.equal(
            actor_target["mlp.0.weight"][:, :75], actor_source["mlp.0.weight"]
        ),
        "actor_new_6_zero": bool(
            torch.count_nonzero(actor_target["mlp.0.weight"][:, 75:81]) == 0
        ),
        "critic_source_shape_is_512x78": tuple(critic_source["mlp.0.weight"].shape)
        == (512, 78),
        "critic_target_shape_is_512x109": tuple(critic_target["mlp.0.weight"].shape)
        == (512, 109),
        "critic_common_75_exact": torch.equal(
            critic_target["mlp.0.weight"][:, :75],
            critic_source["mlp.0.weight"][:, :75],
        ),
        "critic_base_lin_exactly_75_78_to_81_84": torch.equal(
            critic_target["mlp.0.weight"][:, 81:84],
            critic_source["mlp.0.weight"][:, 75:78],
        ),
        "critic_other_new_columns_zero": bool(
            torch.count_nonzero(critic_target["mlp.0.weight"][:, 75:81]) == 0
            and torch.count_nonzero(critic_target["mlp.0.weight"][:, 84:109]) == 0
        ),
        "optimizer_empty_before_target_training": len(
            target["optimizer_state_dict"]["state"]
        )
        == 0,
        "iteration_zero_before_target_training": int(target["iter"]) == 0,
        "curriculum_zero_before_target_training": target["infos"]["env_state"][
            "common_step_counter"
        ]
        == 0,
        "approval_mode_is_unambiguous": bool(
            (formal_source and provenance.get("engineering_smoke_only") is False)
            or (
                not formal_source
                and provenance.get("engineering_smoke_only") is True
                and provenance.get("formal_training_allowed") is False
            )
        ),
        "ppo_proof_schema": ppo.get("schema_version")
        == "v049_v045_warmstart_cpu_ppo_smoke_v1",
        "ppo_proof_binds_same_artifact": ppo.get("warm_start", {})
        .get("checkpoint", {})
        .get("sha256")
        == sha256(artifact),
        "ppo_proof_binds_same_provenance": ppo.get("warm_start", {})
        .get("proof", {})
        .get("sha256")
        == sha256(provenance_path),
        "strict_load_and_one_ppo_update_completed": ppo.get(
            "strict_checkpoint_load_completed"
        )
        is True
        and ppo.get("ppo_update_completed") is True,
        "fresh_training_state_checks_passed": _all_true(
            ppo.get("fresh_training_state_checks")
        ),
        "warmstart_and_smoke_do_not_claim_target_skill": provenance.get(
            "quality_passed"
        )
        is False
        and ppo.get("quality_passed") is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    output = _next_output()
    payload = {
        "schema_version": "v049_v045_warmstart_semantic_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": {"path": str(artifact), "sha256": sha256(artifact)},
        "provenance": {
            "path": str(provenance_path),
            "sha256": sha256(provenance_path),
        },
        "ppo_proof": {"path": str(ppo_path), "sha256": sha256(ppo_path)},
        "source_gate_evidence": {
            "mode": "formal" if formal_source else "engineering_smoke",
            "source_evidence_valid": source_evidence_valid,
            "source_evidence_error": source_evidence_error,
            "training_loader_valid": training_loader_valid,
            "training_loader_error": training_loader_error,
        },
        "checks": checks,
        "warmstart_semantics_passed": passed,
        "formal_v045_product_available": formal_source,
        "formal_training_allowed": bool(formal_source and passed),
        "training_completed": False,
        "fixed_validation_passed": False,
        "heldout_validation_passed": False,
        "quality_passed": False,
        "failed_checks": [name for name, value in checks.items() if not value],
        "conclusion": (
            "Shape-aware loading and one CPU PPO update passed with the requested formal or engineering source gate; receiving skill is not claimed."
            if passed
            else "v049 warm-start audit failed."
        ),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "warmstart_semantics_passed": passed,
                "formal_training_allowed": bool(formal_source and passed),
                "quality_passed": False,
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
