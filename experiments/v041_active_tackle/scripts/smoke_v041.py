from __future__ import annotations

from pathlib import Path

from experiments.v040_pass_receive.runner import load_json, run_episode, unique_output, write_json
from experiments.v041_active_tackle.active_tackle_env import ActiveTackleEnv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_json(ROOT / "configs" / "v041_active_tackle.json")
    env = ActiveTackleEnv(config, seed=41000)
    try:
        row = run_episode(env, seed=41000, maximum_steps=8)
    finally:
        env.close()
    passed = bool(row["finite_state"] and row["all_23_targets_sent"] and not row["fell"])
    payload = {"version": config["version"], "smoke_steps": 8, "quality_passed": passed, "result": row}
    output = unique_output(ROOT / "metrics" / "v041_smoke_8_steps.json")
    write_json(output, payload)
    print(output)
    print(payload)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

