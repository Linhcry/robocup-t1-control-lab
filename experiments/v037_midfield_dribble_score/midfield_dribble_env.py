"""在 v036 真实射门环境上增加连续控球奖励和课程阶段指标。"""

from __future__ import annotations

from copy import deepcopy
from math import atan2
from pathlib import Path
from typing import Any

import numpy as np

from experiments.v036_single_robot_score.single_robot_score_env import SingleRobotScoreEnv


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]


def dribble_teacher_action(observation: np.ndarray) -> np.ndarray:
    """先走到球门反方向的球后方，再沿球门方向穿过球完成连续推进。"""

    obs = np.asarray(observation, dtype=np.float32)
    ball = np.asarray([obs[5] * 3.0, obs[6] * 2.0], dtype=np.float64)
    goal = np.asarray([obs[9] * 3.0, obs[10] * 2.0], dtype=np.float64)
    ball_to_goal = goal - ball
    ball_to_goal /= max(float(np.linalg.norm(ball_to_goal)), 1.0e-6)
    behind_ball = ball - 0.42 * ball_to_goal
    if float(np.linalg.norm(behind_ball)) > 0.22:
        target = behind_ball
        speed = 0.34
    else:
        target = ball + 0.80 * ball_to_goal
        speed = 0.45
    target_angle = float(atan2(target[1], target[0]))
    vx = speed * float(np.clip(np.cos(target_angle), 0.0, 1.0))
    vy = float(np.clip(speed * np.sin(target_angle), -0.20, 0.20))
    return np.asarray(
        [vx / 0.45, vy / 0.20, np.clip(target_angle / 0.45, -1.0, 1.0)],
        dtype=np.float32,
    )


def environment_config_for_stage(config: dict[str, Any], stage_index: int) -> dict[str, Any]:
    """把 v037 的某一级课程转换成 v036 基础环境能够读取的完整配置。"""

    stages = config["stages"]
    if not 0 <= stage_index < len(stages):
        raise IndexError(f"stage must be in [0,{len(stages) - 1}], got {stage_index}")
    stage = stages[stage_index]
    base = config["base_environment"]
    environment = {
        "prepare_seconds": float(base["prepare_seconds"]),
        "action_repeat": int(base["action_repeat"]),
        "episode_seconds": float(stage["episode_seconds"]),
        "player_ball_distance_m": deepcopy(base["player_ball_distance_m"]),
        "ball_goal_distance_m": deepcopy(stage["ball_goal_distance_m"]),
        "ball_y_m": deepcopy(stage["ball_y_m"]),
        "player_y_offset_m": deepcopy(stage["player_y_offset_m"]),
        "player_yaw_rad": deepcopy(stage["player_yaw_rad"]),
    }
    # v038 以后可以按课程阶段提高速度；没有这些键的 v037 仍完全沿用旧默认值。
    for key in (
        "max_vx_mps",
        "max_vy_mps",
        "max_yaw_rate_rps",
        "observation_vx_scale_mps",
        "observation_vy_scale_mps",
    ):
        value = stage.get(key, base.get(key))
        if value is not None:
            environment[key] = float(value)
    return {
        "version": config["version"],
        "environment": environment,
    }


class MidfieldDribbleScoreEnv(SingleRobotScoreEnv):
    """奖励球向球门推进、回到球门中心线，以及触球后持续保持可控距离。"""

    def __init__(self, config: dict[str, Any], stage_index: int, *, seed: int | None = None) -> None:
        self.curriculum_config = config
        self.stage_index = int(stage_index)
        self.stage = config["stages"][self.stage_index]
        self.reward_config = config["reward"]
        super().__init__(environment_config_for_stage(config, self.stage_index), seed=seed)
        self.post_contact_steps = 0
        self.controlled_possession_steps = 0
        self.maximum_distance_after_contact = 0.0
        self.lost_possession = False
        self.initial_goal_distance = 0.0

    def _goal_distance(self) -> float:
        ball = np.asarray(self.simulation.ball.xpos[:2], dtype=np.float64)
        return float(np.linalg.norm(np.asarray([27.5, 0.0], dtype=np.float64) - ball))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        self.post_contact_steps = 0
        self.controlled_possession_steps = 0
        self.maximum_distance_after_contact = 0.0
        self.lost_possession = False
        self.initial_goal_distance = self._goal_distance()
        info.update(self._dribble_info())
        return observation, info

    def _dribble_info(self) -> dict[str, Any]:
        ratio = self.controlled_possession_steps / max(1, self.post_contact_steps)
        return {
            "stage_index": self.stage_index,
            "stage_name": str(self.stage["name"]),
            "initial_goal_distance_m": float(self.initial_goal_distance),
            "remaining_goal_distance_m": float(self._goal_distance()),
            "goal_distance_progress_m": float(self.initial_goal_distance - self._goal_distance()),
            "post_contact_steps": int(self.post_contact_steps),
            "controlled_possession_steps": int(self.controlled_possession_steps),
            "controlled_possession_ratio": float(ratio),
            "maximum_distance_after_contact_m": float(self.maximum_distance_after_contact),
            "lost_possession": bool(self.lost_possession),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        old_abs_lateral = abs(float(self.simulation.ball.xpos[1]))
        old_goal_distance = self._goal_distance()
        observation, reward, terminated, truncated, info = super().step(action)
        new_abs_lateral = abs(float(self.simulation.ball.xpos[1]))
        new_goal_distance = self._goal_distance()
        robot_ball_distance = float(
            np.linalg.norm(
                np.asarray(self.agent.xpos[:2], dtype=np.float64)
                - np.asarray(self.simulation.ball.xpos[:2], dtype=np.float64)
            )
        )

        reward_cfg = self.reward_config
        center_progress = float(np.clip(old_abs_lateral - new_abs_lateral, -0.15, 0.15))
        goal_progress = float(np.clip(old_goal_distance - new_goal_distance, -0.20, 0.20))
        # 父环境已经奖励 x 方向推进；这里的球门距离项会同时考虑横向偏差。
        reward += 2.0 * goal_progress
        reward += float(reward_cfg["center_progress_weight"]) * center_progress
        reward -= float(reward_cfg["lateral_error_weight"]) * min(new_abs_lateral, 8.0)

        if self.contacted_ball:
            self.post_contact_steps += 1
            self.maximum_distance_after_contact = max(self.maximum_distance_after_contact, robot_ball_distance)
            control_distance = float(reward_cfg["control_distance_m"])
            if robot_ball_distance <= control_distance:
                self.controlled_possession_steps += 1
                reward += float(reward_cfg["controlled_possession_bonus"])
            else:
                reward -= float(reward_cfg["lost_possession_weight"]) * min(
                    robot_ball_distance - control_distance,
                    3.0,
                )
            self.lost_possession = self.lost_possession or robot_ball_distance >= float(reward_cfg["lost_distance_m"])

        if self.scored:
            reward += float(reward_cfg["goal_bonus"])
        if truncated and not self.scored:
            reward -= float(reward_cfg["timeout_penalty"])
        info.update(self._dribble_info())
        return observation, float(reward), terminated, truncated, info


class ResidualMidfieldDribbleScoreEnv(MidfieldDribbleScoreEnv):
    """教师保证基本追球路径，PPO动作只负责三个通道的小幅残差修正。"""

    def __init__(self, config: dict[str, Any], stage_index: int, *, seed: int | None = None) -> None:
        super().__init__(config, stage_index, seed=seed)
        self.residual_scale = np.asarray(config["residual_action_scale"], dtype=np.float32)
        if self.residual_scale.shape != (3,):
            raise ValueError("residual_action_scale must contain vx, vy and yaw scales")

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        residual = np.asarray(action, dtype=np.float32)
        if residual.shape != (3,) or not np.isfinite(residual).all():
            raise FloatingPointError("residual action must be a finite three-dimensional vector")
        base_action = dribble_teacher_action(self._observation())
        executed_action = np.clip(base_action + self.residual_scale * residual, -1.0, 1.0)
        observation, reward, terminated, truncated, info = super().step(executed_action)
        info.update(
            {
                "teacher_action": base_action.tolist(),
                "residual_action": residual.tolist(),
                "executed_action": executed_action.tolist(),
            }
        )
        return observation, reward, terminated, truncated, info
