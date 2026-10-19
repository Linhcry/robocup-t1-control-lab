"""Regression tests for v039 data-split and curriculum gates."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.v039_getup_recovery.getup_env import load_config
from experiments.v039_getup_recovery.scripts.evaluate_stage import (
    FIXED_VALIDATION_SEED_START,
    HELD_OUT_TEST_SEED_START,
    resolve_evaluation_request,
)
from experiments.v039_getup_recovery.scripts.train_stage import (
    validate_previous_stage_proof,
)


class EvaluationSplitTests(unittest.TestCase):
    def test_default_splits_use_disjoint_seed_namespaces_and_configured_sizes(self) -> None:
        config = load_config()
        fixed = resolve_evaluation_request(
            held_out=False,
            seed_start=None,
            episodes_per_pose=None,
            config=config,
        )
        held_out = resolve_evaluation_request(
            held_out=True,
            seed_start=None,
            episodes_per_pose=None,
            config=config,
        )
        self.assertEqual(fixed, ("fixed_validation", FIXED_VALIDATION_SEED_START, 20))
        self.assertEqual(held_out, ("held_out_test", HELD_OUT_TEST_SEED_START, 100))
        self.assertGreaterEqual(held_out[1], fixed[1] + fixed[2])

    def test_held_out_rejects_fixed_validation_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            resolve_evaluation_request(
                held_out=True,
                seed_start=FIXED_VALIDATION_SEED_START,
                episodes_per_pose=5,
                config=load_config(),
            )


class PreviousStageProofTests(unittest.TestCase):
    def test_resume_alone_is_not_a_quality_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "stage0.zip"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "previous-stage-proof"):
                validate_previous_stage_proof(stage=1, resume=checkpoint, proof_path=None)

    def test_matching_passed_fixed_validation_proof_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "stage0.zip"
            checkpoint.write_bytes(b"checkpoint")
            proof = root / "fixed.json"
            proof.write_text(
                json.dumps(
                    {
                        "stage": 0,
                        "evaluation_split": "fixed_validation",
                        "checkpoint": str(checkpoint.resolve()),
                        "summary": {"quality_passed": True},
                    }
                ),
                encoding="utf-8",
            )
            result = validate_previous_stage_proof(
                stage=1, resume=checkpoint, proof_path=proof
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["stage"], 0)
            self.assertEqual(len(str(result["sha256"])), 64)

    def test_wrong_stage_or_failed_proof_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "stage0.zip"
            checkpoint.write_bytes(b"checkpoint")
            proof = root / "fixed.json"
            proof.write_text(
                json.dumps(
                    {
                        "stage": 0,
                        "evaluation_split": "fixed_validation",
                        "checkpoint": str(checkpoint.resolve()),
                        "summary": {"quality_passed": False},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "quality_passed"):
                validate_previous_stage_proof(
                    stage=1, resume=checkpoint, proof_path=proof
                )


if __name__ == "__main__":
    unittest.main()
