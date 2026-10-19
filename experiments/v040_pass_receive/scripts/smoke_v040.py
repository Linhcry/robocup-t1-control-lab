from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.v040_pass_receive.pass_receive_env import PassReceiveEnv
from experiments.v040_pass_receive.runner import load_json, run_episode, unique_output, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_json(ROOT / "configs" / "v040_pass_receive.json")
    env = PassReceiveEnv(config, seed=40000)
    try:
        row = run_episode(env, seed=40000, maximum_steps=8)
    finally:
        env.close()
    passed = bool(row["finite_state"] and row["all_23_targets_sent"] and not row["fell"])
    payload = {"version": config["version"], "smoke_steps": 8, "quality_passed": passed, "result": row}
    output = unique_output(ROOT / "metrics" / "v040_smoke_8_steps.json")
    write_json(output, payload)
    print(output)
    print(payload)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

