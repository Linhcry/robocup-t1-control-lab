"""Eight-step physics smoke test plus recovery-supervisor state test."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v039_getup_recovery.artifacts import write_json  # noqa: E402
from experiments.v039_getup_recovery.getup_env import T1GetUpEnv  # noqa: E402
from experiments.v039_getup_recovery.recovery_fsm import (  # noqa: E402
    OfficialRecoverySupervisor,
    RecoveryDecision,
    RecoveryState,
)


def main() -> None:
    env = T1GetUpEnv(stage=0, fixed_pose="near_fall")
    observation, reset_info = env.reset(seed=39010)
    steps: list[dict[str, object]] = []
    for index in range(8):
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(23, dtype=np.float32)
        )
        steps.append(
            {
                "step": index + 1,
                "reward": reward,
                "finite": bool(np.isfinite(observation).all() and info["finite"]),
                "height_m": info["height_m"],
                "upright_cos": info["upright_cos"],
                "all_23_targets": info["all_23_targets"],
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        if terminated or truncated:
            break
    env.close()

    requests: list[str] = []
    supervisor = OfficialRecoverySupervisor(lambda: requests.append("kGetUp"))
    first = supervisor.update(
        state=RecoveryState.FALLEN,
        recovery_available=True,
        robot_mode_index=1,
    )
    duplicate = supervisor.update(
        state=RecoveryState.FALLEN,
        recovery_available=True,
        robot_mode_index=1,
    )
    falling = supervisor.update(
        state=RecoveryState.FALLING,
        recovery_available=True,
        robot_mode_index=1,
    )
    passed = bool(
        len(steps) == 8
        and all(step["finite"] for step in steps)
        and all(step["all_23_targets"] for step in steps)
        and reset_info["model_nu"] == 23
        and first == RecoveryDecision.REQUEST_GET_UP
        and duplicate == RecoveryDecision.WAIT
        and falling == RecoveryDecision.SAFE_STOP
        and requests == ["kGetUp"]
    )
    payload = {
        "experiment": "v039_getup_recovery",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reset": reset_info,
        "steps": steps,
        "fsm": {
            "first": first.value,
            "duplicate": duplicate.value,
            "falling": falling.value,
            "request_count": len(requests),
        },
        "quality_passed": passed,
    }
    output = write_json(
        Path(__file__).resolve().parents[1] / "metrics" / "v039_smoke_8_steps.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved={output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
