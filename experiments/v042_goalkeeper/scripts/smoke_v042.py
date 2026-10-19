from __future__ import annotations

from pathlib import Path

from experiments.v040_pass_receive.runner import load_json, run_episode, unique_output, write_json
from experiments.v042_goalkeeper.goalkeeper_env import GoalkeeperGroundBlockEnv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_json(ROOT / "configs" / "v042_goalkeeper.json")
    env = GoalkeeperGroundBlockEnv(config, seed=42000)
    try:
        row = run_episode(env, seed=42000, maximum_steps=8)
    finally:
        env.close()
    passed = bool(row["finite_state"] and row["all_23_targets_sent"] and not row["fell"])
    payload = {"version": config["version"], "smoke_steps": 8, "quality_passed": passed, "result": row}
    output = unique_output(ROOT / "metrics" / "v042_smoke_8_steps.json")
    write_json(output, payload)
    print(output)
    print(payload)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

