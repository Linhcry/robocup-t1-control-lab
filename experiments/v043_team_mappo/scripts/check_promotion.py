"""Apply all v043 promotion gates to one fixed/held-out evaluation file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v043_team_mappo.artifacts import next_numbered_path
from experiments.v043_team_mappo.promotion import assess_promotion


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    evaluation_path = args.evaluation.expanduser().resolve()
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assessment = assess_promotion(evaluation)
    output = next_numbered_path(EXPERIMENT_ROOT / "metrics", "v043_promotion_gate", ".json")
    payload = {
        "version": "v043_team_mappo",
        "evaluation": str(evaluation_path),
        **assessment,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(output)}, ensure_ascii=False, indent=2))
    if not assessment["quality_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
