"""运行一场上下半场各五分钟的 7v7 足球比赛。

默认双方先使用只读局部观测的启发式策略。以后传入 v035 战术 ``.pt`` 模型时，
七名球员共享网络参数，但每次仍分别用自己的 64 维观测推理。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np

from rcsssmj.games.soccer.game_phase import GamePhase
from rcsssmj.games.soccer.play_mode import PlayMode
from rcsssmj.games.teams import TeamSide

from experiments.v035_7v7_self_play.artifacts import next_numbered_path
from experiments.v035_7v7_self_play.networks import TacticalPolicy
from experiments.v035_7v7_self_play.seven_v_seven_core import SevenVsSevenCore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOW_LEVEL_MODEL = PROJECT_ROOT.parents[1] / "third_party" / "booster_gym" / "deploy" / "models" / "T1.pt"


def run_half(
    *,
    phase: GamePhase,
    left_team_name: str,
    right_team_name: str,
    half_seconds: float,
    prepare_seconds: float,
    low_level_model: Path,
    left_policy: TacticalPolicy,
    right_policy: TacticalPolicy,
    tactical_dt: float,
    viewer_enabled: bool,
    realtime: bool,
    kickoff_side: TeamSide,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    """运行一个半场；第二半场通过反转左右球队名称真正完成换边。"""

    core = SevenVsSevenCore(
        low_level_model,
        left_team_name=left_team_name,
        right_team_name=right_team_name,
        phase=phase,
    )
    core.place_teams()
    core.prepare(prepare_seconds)
    core.kick_off(kickoff_side)

    viewer = None
    if viewer_enabled:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(core.simulation.mj_model, core.simulation.mj_data)

    physics_dt = float(core.simulation.timestep)
    tactical_interval = max(1, round(float(tactical_dt) / physics_dt))
    maximum_steps = max(1, round(float(half_seconds) / physics_dt))
    left_actions = np.zeros((7, 3), dtype=np.float32)
    right_actions = np.zeros((7, 3), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    demonstration_observations: list[np.ndarray] = []
    demonstration_actions: list[np.ndarray] = []
    wall_start = time.perf_counter()
    start_play_time = float(core.simulation.game_state.play_time)
    completed_steps = 0

    try:
        for step in range(maximum_steps):
            step_start = time.perf_counter()
            if step % tactical_interval == 0:
                left_observations = core.observations(1)
                right_observations = core.observations(-1)
                left_actions = left_policy.predict(left_observations)
                right_actions = right_policy.predict(right_observations)
                demonstration_observations.extend([left_observations.copy(), right_observations.copy()])
                demonstration_actions.extend([left_actions.copy(), right_actions.copy()])

            core.physics_step(left_actions, right_actions)
            completed_steps = step + 1
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
            if realtime:
                remaining = physics_dt - (time.perf_counter() - step_start)
                if remaining > 0.0:
                    time.sleep(remaining)

            if step % max(1, round(1.0 / physics_dt)) == 0 or step == maximum_steps - 1:
                state = core.simulation.game_state
                safety = core.safety_snapshot()
                rows.append(
                    {
                        "phase": phase.name,
                        "play_time_s": float(state.play_time),
                        "play_mode": state.play_mode.value,
                        "left_team": left_team_name,
                        "right_team": right_team_name,
                        "left_score": int(state.left_team_score),
                        "right_score": int(state.right_team_score),
                        "ball_x_m": float(core.simulation.ball.xpos[0]),
                        "ball_y_m": float(core.simulation.ball.xpos[1]),
                        "standing_players": int(safety["standing_players"]),
                        "minimum_upright": float(safety["minimum_upright"]),
                    }
                )
            if core.simulation.game_state.play_mode == PlayMode.GAME_OVER:
                break
    finally:
        if viewer is not None:
            viewer.close()

    state = core.simulation.game_state
    result = {
        "phase": phase.name,
        "left_team": left_team_name,
        "right_team": right_team_name,
        "left_score": int(state.left_team_score),
        "right_score": int(state.right_team_score),
        "start_play_time_s": start_play_time,
        "end_play_time_s": float(state.play_time),
        "simulated_play_seconds": completed_steps * physics_dt,
        "physics_steps": completed_steps,
        "wall_seconds": time.perf_counter() - wall_start,
        "final_play_mode": state.play_mode.value,
        "field_runtime_m": list(core.simulation.field.field_dim[:2]),
        "safety": core.safety_snapshot(),
    }
    return result, rows, demonstration_observations, demonstration_actions


def main() -> None:
    """解析命令行，依次运行两半场，并保存比分、时序和模仿学习数据。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-seconds", type=float, default=300.0)
    parser.add_argument("--halves", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prepare-seconds", type=float, default=3.0)
    parser.add_argument("--tactical-dt", type=float, default=0.10)
    parser.add_argument("--low-level-model", type=Path, default=DEFAULT_LOW_LEVEL_MODEL)
    parser.add_argument("--left-tactical-model", type=Path)
    parser.add_argument("--right-tactical-model", type=Path)
    parser.add_argument("--viewer", action="store_true", help="打开 MuJoCo 实时窗口")
    parser.add_argument("--realtime", action="store_true", help="按真实 20ms 周期限速；录像/观看时使用")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "matches")
    args = parser.parse_args()

    left_policy = TacticalPolicy(args.left_tactical_model)
    right_policy = TacticalPolicy(args.right_tactical_model)
    half_specs = [
        (GamePhase.FIRST_HALF, "QiBlue", "QiRed", TeamSide.LEFT),
        # 第二半场重新创建服务器并反转球队连接顺序，裁判计分与实际换边保持一致。
        (GamePhase.SECOND_HALF, "QiRed", "QiBlue", TeamSide.LEFT),
    ][: args.halves]

    half_results: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_observations: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    total_score = {"QiBlue": 0, "QiRed": 0}
    for phase, left_name, right_name, kickoff_side in half_specs:
        result, rows, observations, actions = run_half(
            phase=phase,
            left_team_name=left_name,
            right_team_name=right_name,
            half_seconds=args.half_seconds,
            prepare_seconds=args.prepare_seconds,
            low_level_model=args.low_level_model,
            left_policy=left_policy,
            right_policy=right_policy,
            tactical_dt=args.tactical_dt,
            viewer_enabled=args.viewer,
            realtime=args.realtime,
            kickoff_side=kickoff_side,
        )
        half_results.append(result)
        all_rows.extend(rows)
        all_observations.extend(observations)
        all_actions.extend(actions)
        total_score[left_name] += int(result["left_score"])
        total_score[right_name] += int(result["right_score"])

    summary_path = next_numbered_path(args.output_dir, "match", "_summary.json")
    run_stem = summary_path.name.removesuffix("_summary.json")
    timeseries_path = args.output_dir / f"{run_stem}_timeseries.csv"
    demonstrations_path = args.output_dir / f"{run_stem}_demonstrations.npz"
    payload = {
        "match": "7v7 football",
        "requested_halves": args.halves,
        "seconds_per_half": args.half_seconds,
        "regulation_seconds_when_default": 600,
        "side_change_implemented": args.halves == 2,
        "score": total_score,
        "winner": "draw" if total_score["QiBlue"] == total_score["QiRed"] else max(total_score, key=total_score.get),
        "low_level_model": str(args.low_level_model.resolve()),
        "left_tactical_model": None if args.left_tactical_model is None else str(args.left_tactical_model.resolve()),
        "right_tactical_model": None if args.right_tactical_model is None else str(args.right_tactical_model.resolve()),
        "field_runtime_m": half_results[0]["field_runtime_m"],
        "halves": half_results,
        "timeseries": str(timeseries_path),
        "demonstrations": str(demonstrations_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if all_rows:
        with timeseries_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    np.savez_compressed(
        demonstrations_path,
        observations=np.asarray(all_observations, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.float32),
    )
    print(json.dumps({**payload, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
