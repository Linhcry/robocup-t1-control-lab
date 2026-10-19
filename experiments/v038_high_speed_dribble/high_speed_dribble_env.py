"""v038 高速控球环境：沿用 v037 观测与残差动作，只逐级扩大安全速度。"""

from __future__ import annotations

from math import atan2
from typing import Any

import numpy as np

from experiments.v037_midfield_dribble_score.midfield_dribble_env import MidfieldDribbleScoreEnv


def high_speed_teacher_action(
    observation: np.ndarray,
    *,
    max_vx: float,
    max_vy: float,
    max_yaw_rate: float,
    cruise_speed: float,
    approach_speed: float,
) -> np.ndarray:
    """生成可解释的高速控球基准动作，PPO 只学习其小幅修正量。"""

    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (17,) or not np.isfinite(obs).all():
        raise ValueError(f"v038 teacher expects a finite (17,) observation, got {obs.shape}")
    ball = np.asarray([obs[5] * 3.0, obs[6] * 2.0], dtype=np.float64)
    goal = np.asarray([obs[9] * 3.0, obs[10] * 2.0], dtype=np.float64)
    ball_to_goal = goal - ball
    ball_to_goal /= max(float(np.linalg.norm(ball_to_goal)), 1.0e-6)
    behind_ball = ball - 0.42 * ball_to_goal
    if float(np.linalg.norm(behind_ball)) > 0.22:
        target = behind_ball
        speed = min(float(approach_speed), float(max_vx))
    else:
        target = ball + 1.00 * ball_to_goal
        speed = min(float(cruise_speed), float(max_vx))

    target_angle = float(atan2(target[1], target[0]))
    forward = speed * float(np.clip(np.cos(target_angle), 0.0, 1.0))
    lateral = float(np.clip(speed * np.sin(target_angle), -max_vy, max_vy))
    # 转向动作仍按约 0.45 rad 的方向误差归一化，随后由环境映射到安全角速度。
    yaw_action = float(np.clip(target_angle / 0.45, -1.0, 1.0))
    return np.asarray(
        [forward / max_vx, lateral / max_vy, yaw_action],
        dtype=np.float32,
    )


class HighSpeedResidualDribbleEnv(MidfieldDribbleScoreEnv):
    """在高速几何教师上训练三维残差，并记录真正的受控带球速度。"""

    def __init__(self, config: dict[str, Any], stage_index: int, *, seed: int | None = None) -> None:
        super().__init__(config, stage_index, seed=seed)
        self.residual_scale = np.asarray(config["residual_action_scale"], dtype=np.float32)
        if self.residual_scale.shape != (3,) or not np.isfinite(self.residual_scale).all():
            raise ValueError("residual_action_scale must contain three finite values")
        self.teacher_cruise_speed = float(self.stage["teacher_cruise_speed_mps"])
        self.teacher_approach_speed = float(self.stage["teacher_approach_speed_mps"])
        self.target_controlled_speed = float(self.stage["target_controlled_speed_mps"])
        self.gait_frequency_hz = float(self.stage["gait_frequency_hz"])
        self.sideline_guard_start = float(config["base_environment"]["sideline_guard_start_m"])
        if not 1.0 <= self.gait_frequency_hz <= 2.0:
            raise ValueError("gait_frequency_hz must stay inside the official 1-2 Hz training range")
        # T1.pt 的原始训练课程覆盖 1-2 Hz；v038 逐阶段显式记录所选频率。
        self.controller.policy.gait_frequency = self.gait_frequency_hz
        self.controlled_speed_samples: list[float] = []
        self.maximum_controlled_speed = 0.0
        self.commanded_forward_speed_samples: list[float] = []
        self.sideline_guard_steps = 0
        self.sideline_guard_phase_steps = {"behind_inside": 0, "behind_outside": 0, "push_inward": 0}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        self.controller.policy.gait_frequency = self.gait_frequency_hz
        self.controlled_speed_samples = []
        self.maximum_controlled_speed = 0.0
        self.commanded_forward_speed_samples = []
        self.sideline_guard_steps = 0
        self.sideline_guard_phase_steps = {"behind_inside": 0, "behind_outside": 0, "push_inward": 0}
        info.update(self._speed_info())
        return observation, info

    def _speed_info(self) -> dict[str, Any]:
        mean_speed = float(np.mean(self.controlled_speed_samples)) if self.controlled_speed_samples else 0.0
        mean_command = (
            float(np.mean(self.commanded_forward_speed_samples))
            if self.commanded_forward_speed_samples
            else 0.0
        )
        return {
            "max_vx_mps": float(self.max_vx),
            "gait_frequency_hz": float(self.gait_frequency_hz),
            "target_controlled_speed_mps": float(self.target_controlled_speed),
            "mean_controlled_robot_speed_mps": mean_speed,
            "maximum_controlled_robot_speed_mps": float(self.maximum_controlled_speed),
            "mean_commanded_forward_speed_mps": mean_command,
            "controlled_speed_sample_count": len(self.controlled_speed_samples),
            "sideline_guard_steps": int(self.sideline_guard_steps),
            "sideline_guard_phase_steps": dict(self.sideline_guard_phase_steps),
        }

    def _action_to_world_target(self, target_world_xy: np.ndarray, speed: float) -> np.ndarray:
        """把一个世界坐标路点变成当前 3 维速度动作。"""

        relative_world = np.asarray(target_world_xy, dtype=np.float64) - np.asarray(
            self.agent.xpos[:2], dtype=np.float64
        )
        relative_body = self._body_vector(np.asarray([*relative_world, 0.0], dtype=np.float64))[:2]
        target_angle = float(atan2(relative_body[1], relative_body[0]))
        distance = float(np.linalg.norm(relative_body))
        requested_speed = min(float(speed), self.max_vx) * float(np.clip(distance / 0.8, 0.25, 1.0))
        forward = requested_speed * float(np.clip(np.cos(target_angle), 0.0, 1.0))
        lateral = float(np.clip(requested_speed * np.sin(target_angle), -self.max_vy, self.max_vy))
        return np.asarray(
            [forward / self.max_vx, lateral / self.max_vy, np.clip(target_angle / 0.45, -1.0, 1.0)],
            dtype=np.float32,
        )

    def _teacher_action(self, observation: np.ndarray) -> np.ndarray:
        """先生成普通控球动作；球靠近边线时由安全层把推进方向转回场内。"""

        # 主教师保持 v037 已经验证过的“绕到球后方，再朝球门推进”几何逻辑。
        # 门前多路点漏斗在固定验证中把成功率降到 0%，因此没有进入正式策略。
        teacher = high_speed_teacher_action(
            observation,
            max_vx=self.max_vx,
            max_vy=self.max_vy,
            max_yaw_rate=self.max_yaw_rate,
            cruise_speed=self.teacher_cruise_speed,
            approach_speed=self.teacher_approach_speed,
        )
        ball_world = np.asarray(self.simulation.ball.xpos[:2], dtype=np.float64)
        robot_world = np.asarray(self.agent.xpos[:2], dtype=np.float64)
        ball_world_y = float(self.simulation.ball.xpos[1])

        if abs(ball_world_y) < self.sideline_guard_start:
            return teacher

        self.sideline_guard_steps += 1
        side = float(np.sign(ball_world_y))

        # 机器人若在球的场内一侧，直接追“球后点”必然会先撞球并把它继续推向边线。
        # 因此先从场内一侧退到球后，再绕到场外一侧，最后才朝中路推球。
        behind_x = float(ball_world[0] - 0.75)
        outside_offset = side * 0.70
        inside_offset = -side * 0.70
        is_behind = float(robot_world[0]) <= float(ball_world[0] - 0.50)
        is_outside = side * float(robot_world[1] - ball_world[1]) >= 0.40
        if not is_behind:
            self.sideline_guard_phase_steps["behind_inside"] += 1
            waypoint = np.asarray([behind_x, ball_world[1] + inside_offset], dtype=np.float64)
            return self._action_to_world_target(waypoint, self.teacher_approach_speed)
        if not is_outside:
            self.sideline_guard_phase_steps["behind_outside"] += 1
            waypoint = np.asarray([behind_x, ball_world[1] + outside_offset], dtype=np.float64)
            return self._action_to_world_target(waypoint, self.teacher_approach_speed)

        self.sideline_guard_phase_steps["push_inward"] += 1
        obs = np.asarray(observation, dtype=np.float32)
        ball_body = np.asarray([obs[5] * 3.0, obs[6] * 2.0], dtype=np.float64)
        # 已绕到外侧后，世界坐标仍向对方球门前进，同时给一个更强的回中路分量。
        desired_world = np.asarray([1.0, -2.0 * np.sign(ball_world_y), 0.0], dtype=np.float64)
        desired_body = self._body_vector(desired_world)[:2]
        desired_body /= max(float(np.linalg.norm(desired_body)), 1.0e-6)
        behind_ball = ball_body - 0.42 * desired_body
        if float(np.linalg.norm(behind_ball)) > 0.22:
            target = behind_ball
            speed = min(self.teacher_approach_speed, self.max_vx)
        else:
            target = ball_body + desired_body
            speed = min(0.80 * self.teacher_cruise_speed, self.max_vx)
        target_angle = float(atan2(target[1], target[0]))
        forward = speed * float(np.clip(np.cos(target_angle), 0.0, 1.0))
        lateral = float(np.clip(speed * np.sin(target_angle), -self.max_vy, self.max_vy))
        return np.asarray(
            [forward / self.max_vx, lateral / self.max_vy, np.clip(target_angle / 0.45, -1.0, 1.0)],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        residual = np.asarray(action, dtype=np.float32)
        if residual.shape != (3,) or not np.isfinite(residual).all():
            raise FloatingPointError("v038 residual action must be a finite three-dimensional vector")
        teacher = self._teacher_action(self._observation())
        executed = np.clip(teacher + self.residual_scale * residual, -1.0, 1.0)
        observation, reward, terminated, truncated, info = super().step(executed)

        rotation_world_from_body = np.asarray(self.agent.xmat, dtype=np.float64).reshape(3, 3)
        body_velocity = rotation_world_from_body.T @ np.asarray(self.agent._qvel[:3], dtype=np.float64)
        robot_ball_distance = float(
            np.linalg.norm(
                np.asarray(self.agent.xpos[:2], dtype=np.float64)
                - np.asarray(self.simulation.ball.xpos[:2], dtype=np.float64)
            )
        )
        controlled = self.contacted_ball and robot_ball_distance <= float(self.reward_config["control_distance_m"])
        if controlled:
            forward_speed = max(0.0, float(body_velocity[0]))
            self.controlled_speed_samples.append(forward_speed)
            self.maximum_controlled_speed = max(self.maximum_controlled_speed, forward_speed)
            reward += float(self.reward_config["controlled_speed_weight"]) * float(
                np.clip(forward_speed / max(self.target_controlled_speed, 1.0e-6), 0.0, 1.5)
            )
        self.commanded_forward_speed_samples.append(float(max(0.0, executed[0]) * self.max_vx))
        reward -= float(self.reward_config["decision_time_penalty"])
        # 新随机种子上 v037 的残差曾劣于纯教师；该正则项只保留确实能改善结果的修正。
        reward -= float(self.reward_config["residual_l2_penalty"]) * float(np.square(residual).sum())
        info.update(
            {
                **self._speed_info(),
                "teacher_action": teacher.tolist(),
                "residual_action": residual.tolist(),
                "executed_action": executed.tolist(),
            }
        )
        return observation, float(reward), terminated, truncated, info
