"""供参数共享 IPPO 使用的 7v7 高层训练环境。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rcsssmj.games.soccer.game_phase import GamePhase

from experiments.v035_7v7_self_play.seven_v_seven_core import SevenVsSevenCore
from locomotion.rcss_simulation import torso_upright


class TacticalSelfPlayEnv:
    """左队是学习方、右队是对手；每一步接收两队各自的七个局部动作。"""

    def __init__(
        self,
        low_level_model: str | Path,
        *,
        action_repeat: int = 5,
        episode_high_level_steps: int = 200,
        prepare_seconds: float = 1.0,
        seed: int = 20260718,
    ) -> None:
        self.core = SevenVsSevenCore(
            low_level_model,
            left_team_name="Learner",
            right_team_name="Opponent",
            phase=GamePhase.FIRST_HALF,
        )
        self.action_repeat = int(action_repeat)
        self.episode_high_level_steps = int(episode_high_level_steps)
        self.prepare_seconds = float(prepare_seconds)
        self.rng = np.random.default_rng(seed)
        self.high_level_step = 0
        self.previous_left_score = 0
        self.previous_right_score = 0

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """重置阵型，并在中场附近随机化球的位置形成训练课程。"""

        game_state = self.core.simulation.game_state
        game_state.switch_phase(GamePhase.FIRST_HALF, 0.0)
        game_state.left_team_score = 0
        game_state.right_team_score = 0
        self.core.simulation.referee.reset()
        self.core.place_teams()
        self.core.prepare(self.prepare_seconds)
        ball_x = float(self.rng.uniform(-5.0, 5.0))
        ball_y = float(self.rng.uniform(-3.0, 3.0))
        self.core.simulation.ball.drop_at(ball_x, ball_y)
        mujoco.mj_forward(self.core.simulation.mj_model, self.core.simulation.mj_data)
        self.core.simulation.referee.play_on()
        self.high_level_step = 0
        self.previous_left_score = 0
        self.previous_right_score = 0
        return self.core.observations(1), self.core.observations(-1)

    def step(
        self,
        left_actions: np.ndarray,
        right_actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict[str, Any]]:
        """重复五个 20 ms 底层步，返回七名学习球员各自的奖励。"""

        old_ball_x = float(self.core.simulation.ball.xpos[0])
        old_distances = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(self.core.left_players[number].xpos[:2])
                    - np.asarray(self.core.simulation.ball.xpos[:2])
                )
                for number in range(1, 8)
            ],
            dtype=np.float32,
        )
        for _ in range(self.action_repeat):
            self.core.physics_step(left_actions, right_actions)

        self.high_level_step += 1
        state = self.core.simulation.game_state
        new_ball_x = float(self.core.simulation.ball.xpos[0])
        new_distances = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(self.core.left_players[number].xpos[:2])
                    - np.asarray(self.core.simulation.ball.xpos[:2])
                )
                for number in range(1, 8)
            ],
            dtype=np.float32,
        )

        # 团队共享进球和向前推进奖励；每名球员另有接近球与保持站立的个体奖励。
        left_goal_delta = int(state.left_team_score) - self.previous_left_score
        right_goal_delta = int(state.right_team_score) - self.previous_right_score
        progress_reward = 2.0 * (new_ball_x - old_ball_x)
        goal_reward = 20.0 * float(left_goal_delta - right_goal_delta)
        approach_reward = 0.03 * np.clip(old_distances - new_distances, -0.5, 0.5)
        upright = np.asarray(
            [float(torso_upright(self.core.left_players[number])) for number in range(1, 8)],
            dtype=np.float32,
        )
        standing_reward = 0.01 * np.clip(upright, -1.0, 1.0)
        fall_penalty = np.where(upright < 0.20, -0.5, 0.0).astype(np.float32)
        rewards = (progress_reward + goal_reward + approach_reward + standing_reward + fall_penalty).astype(np.float32)

        self.previous_left_score = int(state.left_team_score)
        self.previous_right_score = int(state.right_team_score)
        done = bool(
            self.high_level_step >= self.episode_high_level_steps
            or left_goal_delta > 0
            or right_goal_delta > 0
        )
        info = {
            "ball_progress_m": new_ball_x - old_ball_x,
            "left_score": int(state.left_team_score),
            "right_score": int(state.right_team_score),
            "mean_upright": float(upright.mean()),
            "fallen_players": int(np.sum(upright < 0.20)),
        }
        return self.core.observations(1), self.core.observations(-1), rewards, done, info

