"""Evaluate a v034 PPO checkpoint with fixed seeds and explicit quality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for path in (PROJECT_ROOT, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stable_baselines3 import PPO  # noqa: E402

from kick_training import (  # noqa: E402
    DEFAULT_CONFIG,
    latest_checkpoint,
    load_config,
    run_fixed_evaluation,
    unique_metrics_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prior-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = None if args.prior_only else (args.model or latest_checkpoint())
    if not args.prior_only and checkpoint is None:
        raise FileNotFoundError("No v034 checkpoint exists; run train_v034.py first")
    policy = None if checkpoint is None else PPO.load(str(checkpoint), device="cpu")
    payload = {
        "model": "phase_prior_only" if checkpoint is None else str(checkpoint),
        "config": str(args.config),
        **run_fixed_evaluation(policy, config),
    }
    stem = "v034_prior_fixed_eval" if checkpoint is None else f"{checkpoint.stem}_fixed_eval"
    output = unique_metrics_path(stem)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

