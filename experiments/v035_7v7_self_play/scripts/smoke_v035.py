"""v035 的 14 台 T1、每台 23 路电机、8 步安全冒烟。"""

from __future__ import annotations

import json
from pathlib import Path

from rcsssmj.games.teams import TeamSide

from experiments.v035_7v7_self_play.artifacts import next_numbered_path
from experiments.v035_7v7_self_play.match_7v7 import DEFAULT_LOW_LEVEL_MODEL, EXPERIMENT_ROOT
from experiments.v035_7v7_self_play.networks import TacticalPolicy
from experiments.v035_7v7_self_play.seven_v_seven_core import SevenVsSevenCore


def main() -> None:
    """构建 14 台机器人并执行恰好 8 个闭环物理步。"""

    core = SevenVsSevenCore(
        DEFAULT_LOW_LEVEL_MODEL,
        left_team_name="QiBlue",
        right_team_name="QiRed",
    )
    core.place_teams()
    core.kick_off(TeamSide.LEFT)
    policy = TacticalPolicy()
    for _ in range(8):
        left_actions = policy.predict(core.observations(1))
        right_actions = policy.predict(core.observations(-1))
        core.physics_step(left_actions, right_actions)

    snapshot = core.safety_snapshot()
    payload = {
        "version": "v035_7v7_self_play",
        "physics_steps": 8,
        "players": 14,
        "motors_per_player": 23,
        "field_m": list(core.simulation.field.field_dim[:2]),
        "half_time_s": int(core.simulation.rules.half_time),
        **snapshot,
    }
    required = (
        payload["finite_qpos"],
        payload["finite_qvel"],
        payload["finite_ctrl"],
        payload["all_14x23_targets_sent"],
        payload["standing_players"] == 14,
    )
    if not all(required):
        raise RuntimeError(json.dumps(payload, ensure_ascii=False, indent=2))
    output = next_numbered_path(EXPERIMENT_ROOT / "metrics", "smoke", ".json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

