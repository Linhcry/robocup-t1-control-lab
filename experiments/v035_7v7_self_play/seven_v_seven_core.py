"""14 台 Booster T1 的内存仿真核心。

此层只负责创建场地、加载底层模型、放置球员、构造局部观测并推进物理。
比赛流程和学习算法在其他文件中，便于逐层阅读和测试。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import mujoco
import numpy as np

from rcsssmj.games.soccer.game_phase import GamePhase
from rcsssmj.games.soccer.soccer_fields import FIFA7vs7SoccerField
from rcsssmj.games.soccer.soccer_rules import SSim2026Rules
from rcsssmj.games.teams import TeamSide

from experiments.v035_7v7_self_play.local_tactics import (
    action_to_motion_command,
    build_local_observation,
    home_position,
)
from locomotion.rcss_simulation import AgentParams, SingleT1SoccerSimulation, torso_upright
from locomotion.rcss_t1_controller import PREPARE_QPOS, RcssT1Controller
from locomotion.t1_spec import clamp_t1_targets


PLAYERS_PER_TEAM: Final[int] = 7


class SevenVsSevenCore:
    """一段半场所需的完整 7v7 物理状态和 14 个独立控制器状态。"""

    def __init__(
        self,
        low_level_model: str | Path,
        *,
        left_team_name: str,
        right_team_name: str,
        phase: GamePhase = GamePhase.FIRST_HALF,
    ) -> None:
        self.low_level_model = Path(low_level_model).expanduser().resolve()
        self.left_team_name = left_team_name
        self.right_team_name = right_team_name
        rules = SSim2026Rules()
        self.simulation = SingleT1SoccerSimulation(
            FIFA7vs7SoccerField(),
            rules,
            initial_game_phase=phase,
            initial_play_time=rules.get_start_time_for(phase),
        )
        if not self.simulation.init():
            raise RuntimeError("RCSSServerMJ 7v7 内存仿真初始化失败")

        self.left_players = self._add_team(left_team_name)
        self.right_players = self._add_team(right_team_name)
        self.controllers = {
            (side_sign, player_no): RcssT1Controller(self.low_level_model)
            for side_sign in (1, -1)
            for player_no in range(1, PLAYERS_PER_TEAM + 1)
        }
        self.policy_time = 0.0
        self.last_left_observations = np.zeros((7, 64), dtype=np.float32)
        self.last_right_observations = np.zeros((7, 64), dtype=np.float32)

    def _add_team(self, team_name: str) -> dict[int, object]:
        """一次加入 1 至 7 号球员，并拒绝缺员的半场。"""

        added = self.simulation.add_players(
            [AgentParams(model_name="T1", team_name=team_name, player_no=no) for no in range(1, 8)]
        )
        if len(added) != PLAYERS_PER_TEAM or any(player is None for player in added):
            raise RuntimeError(f"{team_name} 未能加入完整的 7 名 T1")
        return {int(player.agent_id.player_no): player for player in added if player is not None}

    def place_teams(self) -> None:
        """按左右镜像阵型放置球员，并重置每个策略自己的历史动作。"""

        safe_prepare = clamp_t1_targets(PREPARE_QPOS).astype(np.float64)
        for side_sign, players in ((1, self.left_players), (-1, self.right_players)):
            yaw = 0.0 if side_sign > 0 else np.pi
            for player_no, agent in players.items():
                x, y = home_position(player_no, side_sign)
                agent.drop_at(x, y, yaw)
                agent.init_joints(pos=safe_prepare)
                controller = self.controllers[(side_sign, player_no)]
                controller.policy.reset()
                controller.filtered_targets[:] = PREPARE_QPOS
                controller.target_was_sent.fill(False)

        self.simulation.ball.drop_at(0.0, 0.0)
        self.simulation.mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.simulation.mj_model, self.simulation.mj_data)
        self.policy_time = 0.0

    def prepare(self, seconds: float) -> None:
        """所有球员先保持准备姿态；这段时间处于开球前，不计比赛时钟。"""

        steps = max(0, round(float(seconds) / float(self.simulation.timestep)))
        for _ in range(steps):
            for side_sign, players in ((1, self.left_players), (-1, self.right_players)):
                for player_no, agent in players.items():
                    self.controllers[(side_sign, player_no)].prepare_step(self.simulation, agent)
            self.simulation.step([])

    def kick_off(self, side: TeamSide) -> None:
        """让官方裁判进入开球模式并把球放回中点。"""

        self.simulation.referee.kick_off(side)

    def observations(self, side_sign: int) -> np.ndarray:
        """分别构造七名球员的局部观测；不在球员之间传递私有状态。"""

        players = self.left_players if side_sign > 0 else self.right_players
        opponents = self.right_players if side_sign > 0 else self.left_players
        result = np.stack(
            [
                build_local_observation(
                    self.simulation,
                    players[player_no],
                    players,
                    opponents,
                    side_sign=side_sign,
                )
                for player_no in range(1, 8)
            ]
        )
        if side_sign > 0:
            self.last_left_observations = result
        else:
            self.last_right_observations = result
        return result

    def physics_step(self, left_actions: np.ndarray, right_actions: np.ndarray) -> None:
        """把 14 个三维战术动作送入底层模型，然后只推进一次 20 ms 物理。"""

        team_actions = ((1, self.left_players, left_actions), (-1, self.right_players, right_actions))
        for side_sign, players, actions in team_actions:
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != (7, 3) or not np.isfinite(actions).all():
                raise FloatingPointError("每队动作必须是有限的 (7,3) 数组")
            observations = self.observations(side_sign)
            for index, player_no in enumerate(range(1, 8)):
                command = action_to_motion_command(actions[index], observations[index])
                self.controllers[(side_sign, player_no)].command_step(
                    self.simulation,
                    players[player_no],
                    command,
                    self.policy_time,
                )
        self.simulation.step([])
        self.policy_time += float(self.simulation.timestep)

    def safety_snapshot(self) -> dict[str, float | bool]:
        """返回控制审计和 14 台机器人当前站立情况。"""

        players = list(self.left_players.values()) + list(self.right_players.values())
        heights = np.asarray([float(player.xpos[2]) for player in players])
        uprights = np.asarray([float(torso_upright(player)) for player in players])
        return {
            "finite_qpos": bool(np.isfinite(self.simulation.mj_data.qpos).all()),
            "finite_qvel": bool(np.isfinite(self.simulation.mj_data.qvel).all()),
            "finite_ctrl": bool(np.isfinite(self.simulation.mj_data.ctrl).all()),
            "all_14x23_targets_sent": bool(all(c.all_23_targets_sent for c in self.controllers.values())),
            "minimum_height_m": float(heights.min()),
            "minimum_upright": float(uprights.min()),
            "standing_players": int(np.sum((heights >= 0.34) & (uprights >= 0.20))),
        }

