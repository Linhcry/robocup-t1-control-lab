"""T1 一号门将的横向站位与地滚球封堵课程。"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from locomotion.rcss_simulation import AgentParams, torso_upright

from experiments.v040_pass_receive.ball_event_tracker import BallEventTracker, ContactIdentity
from experiments.v040_pass_receive.skill_core import MultiT1SkillCore, world_to_body


OBSERVATION_DIM = 18
ACTION_DIM = 3
LEFT_GOAL_LINE_X = -27.5
GOAL_HALF_WIDTH = 1.83


class GoalkeeperGroundBlockEnv(gym.Env[np.ndarray, np.ndarray]):
    """随机发射合法地滚球；零动作执行站位教师，PPO 只学习小残差。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        *,
        seed: int | None = None,
        low_level_model: str | Path | None = None,
    ) -> None:
        super().__init__()
        env = config["environment"]
        kwargs = {} if low_level_model is None else {"low_level_model": low_level_model}
        self.core = MultiT1SkillCore(
            [AgentParams(team_name="Goalkeeper", player_no=1)],
            **kwargs,
        )
        self.goalie = self.core.agents[0]
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.prepare_seconds = float(env["prepare_seconds"])
        self.action_repeat = int(env["action_repeat"])
        self.episode_seconds = float(env["episode_seconds"])
        self.shot_speed_range = tuple(float(v) for v in env["shot_speed_mps"])
        self.shot_target_y_range = tuple(float(v) for v in env["shot_target_y_m"])
        self.max_steps = int(ceil(self.episode_seconds / (self.action_repeat * self.core.sim_dt)))
        self.residual_scale = np.asarray(config["residual_action_scale"], dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.tracker = BallEventTracker()
        self.step_count = 0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.shot_target_y = 0.0
        self.initial_ball_speed = 0.0
        self.block_safe_steps = 0
        self.minimum_upright = 1.0
        self.minimum_height = 1.0
        self.ever_fallen = False
        self.success = False
        self.conceded = False

    @property
    def goalie_id(self) -> ContactIdentity:
        return ContactIdentity.from_agent_id(self.goalie.agent_id)  # type: ignore[return-value]

    def _predicted_intercept(self) -> tuple[float, float]:
        ball = np.asarray(self.core.simulation.ball.xpos[:2], dtype=np.float64)
        velocity = np.asarray(self.core.simulation.ball._qvel[:2], dtype=np.float64)
        keeper_x = float(self.goalie.xpos[0])
        if velocity[0] < -0.05:
            time_to_line = max(0.0, (keeper_x - float(ball[0])) / float(velocity[0]))
            predicted_y = float(ball[1] + velocity[1] * time_to_line)
        else:
            time_to_line = 9.0
            predicted_y = float(ball[1])
        return float(np.clip(predicted_y, -1.55, 1.55)), float(np.clip(time_to_line, 0.0, 9.0))

    def teacher_action(self) -> np.ndarray:
        predicted_y, time_to_line = self._predicted_intercept()
        # 身体中心正对球会让地滚球从双脚间穿过；把一只脚而不是身体中心放到球路上。
        foot_offset = -0.18 if predicted_y >= 0.08 else 0.18
        desired_body_y = float(np.clip(predicted_y + foot_offset, -1.45, 1.45))
        error_y = desired_body_y - float(self.goalie.xpos[1])
        lateral = float(np.clip(error_y / 0.60, -1.0, 1.0))
        # 门将保持面向来球，只横移封线；离球很远时降低横移幅度以避免过冲。
        if time_to_line > 5.0:
            lateral *= 0.75
        return np.asarray([0.0, lateral, 0.0], dtype=np.float32)

    def _observation(self) -> np.ndarray:
        rotation = np.asarray(self.goalie.xmat, dtype=np.float64).reshape(3, 3)
        body_velocity = rotation.T @ np.asarray(self.goalie._qvel[:3], dtype=np.float64)
        ball_relative = world_to_body(
            self.goalie,
            np.asarray(self.core.simulation.ball.xpos[:2]) - np.asarray(self.goalie.xpos[:2]),
        )
        ball_velocity = world_to_body(
            self.goalie, np.asarray(self.core.simulation.ball._qvel[:2])
        )
        goal_relative = world_to_body(
            self.goalie,
            np.asarray([LEFT_GOAL_LINE_X, 0.0]) - np.asarray(self.goalie.xpos[:2]),
        )
        predicted_y, time_to_line = self._predicted_intercept()
        observation = np.asarray(
            [
                torso_upright(self.goalie),
                float(self.goalie.xpos[2]),
                np.clip(body_velocity[0] / 0.45, -3.0, 3.0),
                np.clip(body_velocity[1] / 0.24, -3.0, 3.0),
                np.clip(self.goalie._qvel[5] / 2.0, -2.0, 2.0),
                np.clip(ball_relative[0] / 10.0, -2.0, 2.0),
                np.clip(ball_relative[1] / 4.0, -2.0, 2.0),
                np.clip(ball_velocity[0] / 3.0, -2.0, 2.0),
                np.clip(ball_velocity[1] / 3.0, -2.0, 2.0),
                np.clip(goal_relative[0] / 3.0, -2.0, 2.0),
                np.clip(goal_relative[1] / 2.0, -2.0, 2.0),
                predicted_y / GOAL_HALF_WIDTH,
                time_to_line / 9.0,
                float(self.tracker.touched(self.goalie_id)),
                np.clip(self.step_count / max(1, self.max_steps), 0.0, 1.0),
                *self.previous_action.tolist(),
            ],
            dtype=np.float32,
        )
        if observation.shape != (OBSERVATION_DIM,) or not np.isfinite(observation).all():
            raise FloatingPointError("v042 观测必须保持有限且为 18 维")
        return observation

    def _metrics(self) -> dict[str, Any]:
        ball_velocity = np.asarray(self.core.simulation.ball._qvel[:2], dtype=np.float64)
        return {
            "success": bool(self.success),
            "conceded": bool(self.conceded),
            "goalie_touched": bool(self.tracker.touched(self.goalie_id)),
            "contact_body_parts": sorted(
                {
                    part
                    for event in self.tracker.events
                    if event.actor == self.goalie_id
                    for part in event.body_parts
                }
            ),
            "initial_shot_speed_mps": float(self.initial_ball_speed),
            "current_ball_speed_mps": float(np.linalg.norm(ball_velocity)),
            "current_ball_vx_mps": float(ball_velocity[0]),
            "shot_target_y_m": float(self.shot_target_y),
            "goalie_lateral_error_m": float(abs(float(self.goalie.xpos[1]) - self.shot_target_y)),
            "safe_after_contact_seconds": float(
                self.block_safe_steps * self.action_repeat * self.core.sim_dt
            ),
            "minimum_upright": float(self.minimum_upright),
            "minimum_height_m": float(self.minimum_height),
            "fell": bool(self.ever_fallen),
            "episode_steps": int(self.step_count),
            "all_23_targets_sent": bool(self.core.controller_for(self.goalie).all_23_targets_sent),
            "finite_state": bool(
                np.isfinite(self.core.simulation.mj_data.qpos).all()
                and np.isfinite(self.core.simulation.mj_data.qvel).all()
                and np.isfinite(self.core.simulation.mj_data.ctrl).all()
            ),
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
        self.core.reset_players([(-26.15, 0.0, 0.0)], self.prepare_seconds)
        self.shot_target_y = float(self.rng.uniform(*self.shot_target_y_range))
        ball_x = float(self.rng.uniform(-20.5, -18.5))
        # 第一关使用直线地滚球；带横向旋转的弧线球必须在直线封堵通过后再训练。
        ball_y = self.shot_target_y
        self.initial_ball_speed = float(self.rng.uniform(*self.shot_speed_range))
        direction = np.asarray([-1.0, 0.0], dtype=np.float64)
        self.core.simulation.ball.drop_at(ball_x, ball_y)
        self.core.simulation.ball._qvel[:3] = np.asarray(
            [self.initial_ball_speed * direction[0], self.initial_ball_speed * direction[1], 0.0],
            dtype=np.float64,
        )
        mujoco.mj_forward(self.core.simulation.mj_model, self.core.simulation.mj_data)
        self.core.simulation.referee.play_on()
        self.tracker.reset()
        self.step_count = 0
        self.previous_action.fill(0.0)
        self.block_safe_steps = 0
        self.minimum_upright = torso_upright(self.goalie)
        self.minimum_height = float(self.goalie.xpos[2])
        self.ever_fallen = False
        self.success = False
        self.conceded = False
        return self._observation(), self._metrics()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        residual = np.asarray(action, dtype=np.float32)
        if residual.shape != (ACTION_DIM,) or not np.isfinite(residual).all():
            raise FloatingPointError("v042 动作必须是有限的三维数组")
        residual = np.clip(residual, -1.0, 1.0)
        teacher = self.teacher_action()
        executed = np.clip(teacher + self.residual_scale * residual, -1.0, 1.0)
        old_lateral_error = abs(float(self.goalie.xpos[1]) - self.shot_target_y)
        old_right_score = int(self.core.simulation.game_state.right_team_score)
        for _ in range(self.action_repeat):
            self.core.physics_step([executed])
            self.tracker.update(self.core.simulation.ball, float(self.core.simulation.mj_data.time))
            if int(self.core.simulation.game_state.right_team_score) > old_right_score:
                self.conceded = True
                break

        self.step_count += 1
        self.previous_action[:] = residual
        upright = torso_upright(self.goalie)
        height = float(self.goalie.xpos[2])
        self.minimum_upright = min(self.minimum_upright, upright)
        self.minimum_height = min(self.minimum_height, height)
        self.ever_fallen = self.ever_fallen or height < 0.34 or upright < 0.20
        ball_velocity = np.asarray(self.core.simulation.ball._qvel[:2], dtype=np.float64)
        if self.tracker.touched(self.goalie_id) and (
            float(ball_velocity[0]) > 0.05 or float(np.linalg.norm(ball_velocity)) < 0.25
        ):
            self.block_safe_steps += 1
        else:
            # Safety time must be consecutive.  Once a previously touched ball is
            # again travelling dangerously toward the goal, old safe frames cannot
            # be retained and later combined into a false 0.3-second block.
            self.block_safe_steps = 0

        gates = self.config["quality_gates"]
        self.success = bool(
            self.tracker.touched(self.goalie_id)
            and self.block_safe_steps * self.action_repeat * self.core.sim_dt
            >= float(gates["minimum_safe_after_contact_seconds"])
            and not self.conceded
            and not self.ever_fallen
        )
        new_lateral_error = abs(float(self.goalie.xpos[1]) - self.shot_target_y)
        reward = 1.2 * float(np.clip(old_lateral_error - new_lateral_error, -0.2, 0.2))
        reward += 0.08 if self.tracker.touched(self.goalie_id) else 0.0
        reward += 40.0 if self.success else 0.0
        reward -= 40.0 if self.conceded else 0.0
        reward -= 25.0 if self.ever_fallen else 0.0
        reward -= 0.005 * float(np.square(residual).sum())
        metrics = self._metrics()
        metrics.update(
            {
                "teacher_action": teacher.tolist(),
                "residual_action": residual.tolist(),
                "executed_action": executed.tolist(),
            }
        )
        terminated = bool(self.success or self.conceded or self.ever_fallen)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        return self._observation(), float(reward), terminated, truncated, metrics

    def close(self) -> None:
        return None
