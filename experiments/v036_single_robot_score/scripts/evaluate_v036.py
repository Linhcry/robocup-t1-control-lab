"""评估 v036 最新模型或未学习的 v033 追球教师。"""

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

from experiments.v036_single_robot_score.training import (  # noqa: E402
    DEFAULT_CONFIG,
    evaluate_policy,
    latest_checkpoint,
    load_config,
    unique_metrics_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--teacher", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = None if args.teacher else (args.model or latest_checkpoint())
    if checkpoint is None and not args.teacher:
        raise FileNotFoundError("no v036 checkpoint exists; train v036 first")
    policy = None if checkpoint is None else PPO.load(str(checkpoint), device="cpu")
    evaluation = evaluate_policy(policy, config, use_teacher=args.teacher)
    payload = {
        "model": "v033_teacher" if args.teacher else str(checkpoint),
        "config": str(args.config),
        **evaluation,
    }
    stem = "v036_teacher_fixed_eval" if args.teacher else f"{Path(checkpoint).stem}_fixed_eval"
    output = unique_metrics_path(stem)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
