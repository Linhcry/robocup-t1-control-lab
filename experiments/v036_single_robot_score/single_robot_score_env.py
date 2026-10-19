"""真实 RCSSServerMJ 单机器人追球射门环境。

Actor 只输出三个归一化动作：前进速度、横移速度、转向角速度。官方 T1.pt
继续负责 47 维底层观测到 12 个腿关节动作，RcssT1Controller 再发送全部 23 路电机。
"""

from __future__ import annotations

from math import atan2, ceil, pi
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from rcsssmj.games.soccer.game_phase import GamePhase

from locomotion.official_t1_policy import MotionCommand
from locomotion.rcss_simulation import make_single_t1_simulation, torso_upright
from locomotion.rcss_t1_controller import PREPARE_QPOS, RcssT1Controller
from locomotion.t1_spec import clamp_t1_targets


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_LOW_LEVEL_MODEL = PROJECT_ROOT.parents[1] / "third_party" / "booster_gym" / "deploy" / "models" / "T1.pt"

OBSERVATION_DIM = 17
ACTION_DIM = 3
MAX_VX = 0.45
MAX_VY = 0.20
MAX_YAW_RATE = 0.20
RIGHT_GOAL_LINE_X = 27.5
RIGHT_GOAL_BACK_X = 28.5
GOAL_HALF_WIDTH = 1.83
GOAL_HEIGHT = 1.83


def teacher_action(observation: np.ndarray) -> np.ndarray:
    """模仿 v033 已验收追球逻辑，给行为克隆提供三维监督标签。"""

    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (OBSERVATION_DIM,):
        raise ValueError(f"v036 observation must be ({OBSERVATION_DIM},), got {obs.shape}")
    ball_x = float(obs[5] * 3.0)
    ball_y = float(obs[6] * 2.0)
    distance = float(np.hypot(ball_x, ball_y))
    angle = float(atan2(ball_y, ball_x))
    requested_speed = min(MAX_VX, 0.28 + 0.20 * float(obs[13]))
    forward = requested_speed * max(0.25, float(np.cos(angle)))
    lateral = float(np.clip(requested_speed * np.sin(angle), -MAX_VY, MAX_VY))
    if distance < 0.22:
        forward = MAX_VX
    return np.asarray(
        [forward / MAX_VX, lateral / MAX_VY, np.clip(angle / 0.8, -0.35, 0.35)],
        dtype=np.float32,
    )


class SingleRobotScoreEnv(gym.Env[np.ndarray, np.ndarray]):
    """一台左队 T1 从球后方出发，在十秒内追球并攻入右侧球门。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        *,
        seed: int | None = None,
        low_level_model: str | Path = DEFAULT_LOW_LEVEL_MODEL,
    ) -> None:
        super().__init__()
        env_cfg = config["environment"]
        self.config = config
        self.prepare_seconds = float(env_cfg["prepare_seconds"])
        self.action_repeat = int(env_cfg["action_repeat"])
        self.episode_seconds = float(env_cfg["episode_seconds"])
        self.player_ball_distance_range = tuple(float(v) for v in env_cfg["player_ball_distance_m"])
        self.ball_goal_distance_range = tuple(float(v) for v in env_cfg["ball_goal_distance_m"])
        self.ball_y_range = tuple(float(v) for v in env_cfg["ball_y_m"])
        self.player_y_offset_range = tuple(float(v) for v in env_cfg["player_y_offset_m"])
        self.player_yaw_range = tuple(float(v) for v in env_cfg["player_yaw_rad"])
        # 旧版本没有这三个配置项，因此默认值保持 v036/v037 的已验收速度不变。
        # 新课程可以逐级提高上限，而不需要复制整套物理环境代码。
        self.max_vx = float(env_cfg.get("max_vx_mps", MAX_VX))
        self.max_vy = float(env_cfg.get("max_vy_mps", MAX_VY))
        self.max_yaw_rate = float(env_cfg.get("max_yaw_rate_rps", MAX_YAW_RATE))
        command_limits = np.asarray([self.max_vx, self.max_vy, self.max_yaw_rate], dtype=np.float64)
        if not np.isfinite(command_limits).all() or np.any(command_limits <= 0.0):
            raise ValueError("速度与转向上限必须是有限正数")
        # 观测缩放默认固定为旧模型训练时的尺度，扩大动作上限时不改变 v037 的输入分布。
        self.observation_vx_scale = float(env_cfg.get("observation_vx_scale_mps", MAX_VX))
        self.observation_vy_scale = float(env_cfg.get("observation_vy_scale_mps", MAX_VY))
        observation_scales = np.asarray(
            [self.observation_vx_scale, self.observation_vy_scale], dtype=np.float64
        )
        if not np.isfinite(observation_scales).all() or np.any(observation_scales <= 0.0):
            raise ValueError("速度观测缩放必须是有限正数")
        self.rng = np.random.default_rng(seed)

        self.controller = RcssT1Controller(low_level_model)
        self.simulation, self.agent = make_single_t1_simulation()
        self.sim_dt = float(self.simulation.timestep)
        self.decision_dt = self.sim_dt * self.action_repeat
        self.max_episode_steps = int(ceil(self.episode_seconds / self.decision_dt))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBSERVATION_DIM,), dtype=np.float32)

        self.policy_time = 0.0
        self.step_count = 0
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.initial_ball_position = np.zeros(3, dtype=np.float64)
        self.initial_ball_distance = 0.0
        self.minimum_ball_distance = float("inf")
        self.maximum_ball_speed = 0.0
        self.minimum_height = float("inf")
        self.minimum_upright = float("inf")
        self.scored = False
        self.official_goal = False
        self.contacted_ball = False
        self.fallen = False

    def _body_vector(self, world_vector: np.ndarray) -> np.ndarray:
        rotation_world_from_body = np.asarray(self.agent.xmat, dtype=np.float64).reshape(3, 3)
        return rotation_world_from_body.T @ np.asarray(world_vector, dtype=np.float64)

    def _observation(self) -> np.ndarray:
        rotation_world_from_body = np.asarray(self.agent.xmat, dtype=np.float64).reshape(3, 3)
        body_velocity = rotation_world_from_body.T @ np.asarray(self.agent._qvel[:3], dtype=np.float64)
        ball_relative = self._body_vector(
            np.asarray(self.simulation.ball.xpos, dtype=np.float64) - np.asarray(self.agent.xpos, dtype=np.float64)
        )
        ball_velocity = self._body_vector(np.asarray(self.simulation.ball._qvel[:3], dtype=np.float64))
        goal_relative = self._body_vector(
            np.asarray([RIGHT_GOAL_LINE_X, 0.0, 0.0], dtype=np.float64)
            - np.asarray(self.agent.xpos, dtype=np.float64)
        )
        ball_angle = float(atan2(ball_relative[1], ball_relative[0]))
        observation = np.asarray(
            [
                float(torso_upright(self.agent)),
                float(self.agent.xpos[2] / 1.0),
                float(np.clip(body_velocity[0] / self.observation_vx_scale, -3.0, 3.0)),
                float(np.clip(body_velocity[1] / self.observation_vy_scale, -3.0, 3.0)),
                float(np.clip(self.agent._qvel[5] / 2.0, -2.0, 2.0)),
                float(np.clip(ball_relative[0] / 3.0, -2.0, 2.0)),
                float(np.clip(ball_relative[1] / 2.0, -2.0, 2.0)),
                float(np.clip(ball_velocity[0] / 3.0, -2.0, 2.0)),
                float(np.clip(ball_velocity[1] / 3.0, -2.0, 2.0)),
                float(np.clip(goal_relative[0] / 3.0, -2.0, 2.0)),
                float(np.clip(goal_relative[1] / 2.0, -2.0, 2.0)),
                float(np.clip(np.linalg.norm(ball_relative[:2]) / 3.0, 0.0, 2.0)),
                float(np.clip(ball_angle / pi, -1.0, 1.0)),
                float(np.clip(self.step_count / self.max_episode_steps, 0.0, 1.0)),
                *self.previous_action.tolist(),
            ],
            dtype=np.float32,
        )
        if observation.shape != (OBSERVATION_DIM,) or not np.isfinite(observation).all():
            raise FloatingPointError("v036 observation must remain finite and 17-dimensional")
        return observation

    def _ball_is_inside_right_goal(self) -> bool:
        x, y, z = (float(v) for v in self.simulation.ball.xpos)
        return bool(
            RIGHT_GOAL_LINE_X <= x <= RIGHT_GOAL_BACK_X + 0.15
            and abs(y) <= GOAL_HALF_WIDTH
            and 0.0 <= z <= GOAL_HEIGHT
        )

    def _current_info(self) -> dict[str, Any]:
        ball_position = np.asarray(self.simulation.ball.xpos, dtype=np.float64)
        return {
            "scored": bool(self.scored),
            "official_goal": bool(self.official_goal),
            "contacted_ball": bool(self.contacted_ball),
            "fell": bool(self.fallen),
            "episode_steps": int(self.step_count),
            "episode_seconds": float(self.step_count * self.decision_dt),
            "initial_ball_distance_m": float(self.initial_ball_distance),
            "minimum_ball_distance_m": float(self.minimum_ball_distance),
            "ball_forward_m": float(ball_position[0] - self.initial_ball_position[0]),
            "ball_lateral_m": float(ball_position[1] - self.initial_ball_position[1]),
            "maximum_ball_speed_mps": float(self.maximum_ball_speed),
            "minimum_robot_height_m": float(self.minimum_height),
            "minimum_robot_upright": float(self.minimum_upright),
            "left_team_score": int(self.simulation.game_state.left_team_score),
            "all_23_targets_sent": bool(self.controller.all_23_targets_sent),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        game_state = self.simulation.game_state
        game_state.switch_phase(GamePhase.FIRST_HALF, 0.0)
        game_state.left_team_score = 0
        game_state.right_team_score = 0
        self.simulation.referee.reset()
        self.controller.reset_robot(self.simulation, self.agent)

        ball_x = RIGHT_GOAL_LINE_X - float(self.rng.uniform(*self.ball_goal_distance_range))
        ball_y = float(self.rng.uniform(*self.ball_y_range))
        player_x = ball_x - float(self.rng.uniform(*self.player_ball_distance_range))
        player_y = ball_y + float(self.rng.uniform(*self.player_y_offset_range))
        player_yaw = float(self.rng.uniform(*self.player_yaw_range))
        self.agent.drop_at(player_x, player_y, player_yaw)
        self.agent.init_joints(pos=clamp_t1_targets(PREPARE_QPOS).astype(np.float64))
        self.controller.policy.reset()
        self.controller.filtered_targets[:] = PREPARE_QPOS
        self.controller.target_was_sent.fill(False)
        self.simulation.ball.drop_at(ball_x, ball_y)
        self.simulation.mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.simulation.mj_model, self.simulation.mj_data)

        for _ in range(round(self.prepare_seconds / self.sim_dt)):
            self.controller.prepare_step(self.simulation, self.agent)
            self.simulation.step([])

        self.simulation.ball.drop_at(ball_x, ball_y)
        mujoco.mj_forward(self.simulation.mj_model, self.simulation.mj_data)
        self.simulation.referee.play_on()
        self.policy_time = 0.0
        self.step_count = 0
        self.previous_action.fill(0.0)
        self.initial_ball_position = np.asarray(self.simulation.ball.xpos, dtype=np.float64).copy()
        self.initial_ball_distance = float(
            np.linalg.norm(np.asarray(self.agent.xpos[:2]) - self.initial_ball_position[:2])
        )
        self.minimum_ball_distance = self.initial_ball_distance
        self.maximum_ball_speed = 0.0
        self.minimum_height = float(self.agent.xpos[2])
        self.minimum_upright = float(torso_upright(self.agent))
        self.scored = False
        self.official_goal = False
        self.contacted_ball = False
        self.fallen = False
        return self._observation(), self._current_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (ACTION_DIM,) or not np.isfinite(values).all():
            raise FloatingPointError("v036 action must be a finite three-dimensional vector")
        values = np.clip(values, -1.0, 1.0)
        old_ball = np.asarray(self.simulation.ball.xpos, dtype=np.float64).copy()
        old_distance = float(np.linalg.norm(np.asarray(self.agent.xpos[:2]) - old_ball[:2]))
        old_score = int(self.simulation.game_state.left_team_score)

        for _ in range(self.action_repeat):
            ball_relative = self._body_vector(
                np.asarray(self.simulation.ball.xpos, dtype=np.float64)
                - np.asarray(self.agent.xpos, dtype=np.float64)
            )
            ball_angle = float(atan2(ball_relative[1], ball_relative[0]))
            command = MotionCommand(
                vx=float(self.max_vx * values[0]),
                vy=float(self.max_vy * values[1]),
                yaw_rate=float(self.max_yaw_rate * values[2]),
                head_yaw=float(np.clip(ball_angle, -0.8, 0.8)),
                head_pitch=0.28,
            )
            self.controller.command_step(self.simulation, self.agent, command, self.policy_time)
            self.simulation.step([])
            self.policy_time += self.sim_dt
            self.official_goal = self.official_goal or int(self.simulation.game_state.left_team_score) > old_score
            self.scored = self.scored or self.official_goal or self._ball_is_inside_right_goal()
            if self.scored:
                break

        self.step_count += 1
        self.previous_action[:] = values
        new_ball = np.asarray(self.simulation.ball.xpos, dtype=np.float64).copy()
        new_distance = float(np.linalg.norm(np.asarray(self.agent.xpos[:2]) - new_ball[:2]))
        ball_speed = float(np.linalg.norm(self.simulation.ball._qvel[:3]))
        upright = float(torso_upright(self.agent))
        height = float(self.agent.xpos[2])
        self.minimum_ball_distance = min(self.minimum_ball_distance, new_distance)
        self.maximum_ball_speed = max(self.maximum_ball_speed, ball_speed)
        self.minimum_height = min(self.minimum_height, height)
        self.minimum_upright = min(self.minimum_upright, upright)
        ball_displacement = float(np.linalg.norm(new_ball[:2] - self.initial_ball_position[:2]))
        self.contacted_ball = self.contacted_ball or ball_displacement > 0.04 or ball_speed > 0.20
        self.fallen = self.fallen or height < 0.34 or upright < 0.20

        approach_progress = float(np.clip(old_distance - new_distance, -0.15, 0.15))
        ball_progress = float(np.clip(new_ball[0] - old_ball[0], -0.20, 0.20))
        lateral_error = abs(float(new_ball[1])) / GOAL_HALF_WIDTH
        reward = 5.0 * approach_progress + 15.0 * ball_progress
        reward += 0.02 * float(np.clip(upright, -1.0, 1.0))
        reward -= 0.002 * float(np.square(values).sum())
        reward -= 0.005 * lateral_error
        if self.contacted_ball:
            reward += 0.01
        if self.scored:
            reward += 100.0
        if self.fallen:
            reward -= 25.0

        terminated = bool(self.scored or self.fallen)
        truncated = bool(self.step_count >= self.max_episode_steps and not terminated)
        return self._observation(), float(reward), terminated, truncated, self._current_info()

    def close(self) -> None:
        return None
