from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from experiments.v040_pass_receive.runner import basic_summary, evaluate, load_json, unique_output, write_json
from experiments.v041_active_tackle.active_tackle_env import ActiveTackleEnv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--held-out", action="store_true")
    args = parser.parse_args()
    config = load_json(ROOT / "configs" / "v041_active_tackle.json")
    seed_key = "held_out_test_seeds" if args.held_out else "evaluation_seeds"
    seeds = [int(seed) for seed in config[seed_key]]
    model = None if args.model is None else PPO.load(args.model, device="cpu")
    rows = evaluate(lambda seed: ActiveTackleEnv(config, seed=seed), seeds, model=model)
    summary = basic_summary(rows)
    summary.update(
        {
            "turnover_rate": float(np.mean([bool(row["turnover"]) for row in rows])),
            "attacker_touch_rate": float(np.mean([bool(row["attacker_touched"]) for row in rows])),
            "defender_touch_rate": float(np.mean([bool(row["defender_touched"]) for row in rows])),
            "upper_body_contact_rate": float(
                np.mean([int(row["upper_body_contacts"]) > 0 for row in rows])
            ),
            "foul_rate": float(np.mean([bool(row["foul_detected"]) for row in rows])),
            "upper_body_collision_rate": float(
                np.mean([int(row["upper_body_collision_frames"]) > 0 for row in rows])
            ),
            "mean_clearance_m": float(
                np.mean([float(row["clearance_away_from_own_goal_m"]) for row in rows])
            ),
        }
    )
    gates = config["quality_gates"]
    checks = {
        "success_rate": summary["success_rate"] >= float(gates["minimum_success_rate"]),
        "fall_rate": summary["fall_rate"] <= float(gates["maximum_fall_rate"]),
        "upper_body_contact_rate": summary["upper_body_contact_rate"]
        <= float(gates["maximum_upper_body_contact_rate"]),
        "foul_rate": summary["foul_rate"] <= float(gates["maximum_foul_rate"]),
        "finite_state": summary["finite_state_rate"] == 1.0,
        "all_23_targets": summary["all_23_targets_sent_rate"] == 1.0,
    }
    payload = {
        "version": config["version"],
        "policy": "zero residual intercept teacher" if args.model is None else str(args.model),
        "summary": summary,
        "gate_checks": checks,
        "quality_passed": bool(all(checks.values())),
        "episodes": rows,
    }
    if args.held_out:
        name = "v041_teacher_held_out_test.json" if args.model is None else "v041_policy_held_out_test.json"
    else:
        name = "v041_teacher_fixed_eval.json" if args.model is None else "v041_policy_fixed_eval.json"
    output = unique_output(ROOT / "metrics" / name)
    write_json(output, payload)
    print(output)
    print(summary)


if __name__ == "__main__":
    main()
