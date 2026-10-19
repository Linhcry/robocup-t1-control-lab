"""独立加载并评估 v037-r2 指定阶段的最新模型。"""

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

from experiments.v037_midfield_dribble_score.curriculum import DEFAULT_CONFIG, load_config, unique_metrics_path  # noqa: E402
from experiments.v037_midfield_dribble_score.residual_curriculum import (  # noqa: E402
    evaluate_residual_stage,
    latest_residual_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.model or latest_residual_checkpoint(args.stage)
    if checkpoint is None:
        raise FileNotFoundError(f"residual stage {args.stage} has no checkpoint")
    model = PPO.load(str(checkpoint), device="cpu")
    payload = {
        "model": str(checkpoint),
        "config": str(args.config),
        **evaluate_residual_stage(model, config, args.stage),
    }
    metrics = unique_metrics_path(f"{Path(checkpoint).stem}_fixed_eval")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(metrics)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
