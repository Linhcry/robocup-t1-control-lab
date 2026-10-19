"""独立加载 v038 模型，并与纯教师零残差在同一批种子上做消融评估。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from experiments.v038_high_speed_dribble.curriculum import (  # noqa: E402
    DEFAULT_CONFIG,
    evaluate_stage,
    latest_stage_checkpoint,
    load_config,
    unique_metrics_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--held-out-test", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    stage = config["stages"][args.stage]
    seeds = stage["held_out_test_seeds"] if args.held_out_test else stage["evaluation_seeds"]

    checkpoint: Path | None = None
    model: PPO | None = None
    if not args.teacher_only:
        checkpoint = args.model or latest_stage_checkpoint(args.stage)
        if checkpoint is None:
            raise FileNotFoundError(f"v038 stage {args.stage} has no checkpoint")
        model = PPO.load(str(checkpoint), device="cpu")

    payload = {
        "model": "zero_residual_teacher" if checkpoint is None else str(checkpoint),
        "config": str(args.config),
        "split": "held_out_test" if args.held_out_test else "validation",
        **evaluate_stage(model, config, args.stage, seeds=[int(seed) for seed in seeds]),
    }
    source_stem = "teacher" if checkpoint is None else Path(checkpoint).stem
    split_stem = "held_out_test" if args.held_out_test else "fixed_eval"
    metrics = unique_metrics_path(f"{source_stem}_{split_stem}")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.summary_only:
        output = {
            "model": payload["model"],
            "split": payload["split"],
            "summary": payload["summary"],
            "gate_results": payload["gate_results"],
            "quality_passed": payload["quality_passed"],
            "metrics": str(metrics),
        }
    else:
        output = {**payload, "metrics": str(metrics)}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
