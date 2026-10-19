from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from experiments.v040_pass_receive.runner import basic_summary, evaluate, load_json, unique_output, write_json
from experiments.v042_goalkeeper.goalkeeper_env import GoalkeeperGroundBlockEnv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--held-out", action="store_true")
    args = parser.parse_args()
    config = load_json(ROOT / "configs" / "v042_goalkeeper.json")
    seed_key = "held_out_test_seeds" if args.held_out else "evaluation_seeds"
    seeds = [int(seed) for seed in config[seed_key]]
    model = None if args.model is None else PPO.load(args.model, device="cpu")
    rows = evaluate(lambda seed: GoalkeeperGroundBlockEnv(config, seed=seed), seeds, model=model)
    summary = basic_summary(rows)
    summary.update(
        {
            "block_rate": summary["success_rate"],
            "concede_rate": float(np.mean([bool(row["conceded"]) for row in rows])),
            "goalie_touch_rate": float(np.mean([bool(row["goalie_touched"]) for row in rows])),
            "mean_lateral_error_m": float(
                np.mean([float(row["goalie_lateral_error_m"]) for row in rows])
            ),
        }
    )
    gates = config["quality_gates"]
    checks = {
        "block_rate": summary["block_rate"] >= float(gates["minimum_block_rate"]),
        "concede_rate": summary["concede_rate"] <= float(gates["maximum_concede_rate"]),
        "fall_rate": summary["fall_rate"] <= float(gates["maximum_fall_rate"]),
        "finite_state": summary["finite_state_rate"] == 1.0,
        "all_23_targets": summary["all_23_targets_sent_rate"] == 1.0,
    }
    # These checks cover only curriculum stage 0: a narrow, straight ground-ball
    # blocking drill.  They are deliberately kept separate from the competitive
    # goalkeeper gate, which also requires the unimplemented angle/speed grid,
    # diving, recovery, distribution and defender-coordination stages.  A stage-0
    # score must never be auto-selected as a complete goalkeeper policy.
    stage_quality_passed = bool(all(checks.values()))
    curriculum_stages = list(config.get("curriculum_stages", []))
    all_curriculum_stages_implemented = bool(curriculum_stages) and all(
        bool(stage.get("implemented", False)) for stage in curriculum_stages
    )
    competitive_checks = {
        "stage_0_ground_block": stage_quality_passed,
        "all_curriculum_stages_implemented": all_curriculum_stages_implemented,
        "held_out_shot_count_at_least_300": bool(args.held_out and len(rows) >= 300),
        "getup_handoff_validated": False,
        "safe_distribution_validated": False,
        "defender_coordination_validated": False,
    }
    payload = {
        "version": config["version"],
        "evaluation_scope": "stage_0_narrow_straight_ground_ball_blocking",
        "policy": "zero residual lateral positioning teacher" if args.model is None else str(args.model),
        "summary": summary,
        "stage_gate_checks": checks,
        "stage_quality_passed": stage_quality_passed,
        "competitive_gate_checks": competitive_checks,
        "quality_passed": bool(all(competitive_checks.values())),
        "promotion_blocked_by": [
            name for name, passed in competitive_checks.items() if not passed
        ],
        "episodes": rows,
    }
    if args.held_out:
        name = "v042_teacher_held_out_test.json" if args.model is None else "v042_policy_held_out_test.json"
    else:
        name = "v042_teacher_fixed_eval.json" if args.model is None else "v042_policy_fixed_eval.json"
    output = unique_output(ROOT / "metrics" / name)
    write_json(output, payload)
    print(output)
    print(summary)


if __name__ == "__main__":
    main()
