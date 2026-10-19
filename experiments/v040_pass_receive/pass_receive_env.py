"""两台同队 T1 的 2v0 传球、接球与控球课程。"""

from __future__ import annotations

from math import ceil, pi
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from locomotion.rcss_simulation import AgentParams, torso_upright

from experiments.v040_pass_receive.ball_event_tracker import BallEventTracker, ContactIdentity
from experiments.v040_pass_receive.skill_core import MultiT1SkillCore, go_to_world, world_to_body


OBSERVATION_DIM = 34
ACTION_DIM = 6


class PassReceiveEnv(gym.Env[np.ndarray, np.ndarray]):
    """零动作执行几何教师，PPO 动作只学习两名球员各自的三维残差。"""

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
            [
                AgentParams(team_name="PassingTeam", player_no=1),
                AgentParams(team_name="PassingTeam", player_no=2),
            ],
            **kwargs,
        )
        self.passer, self.receiver = self.core.agents
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.prepare_seconds = float(env["prepare_seconds"])
        self.action_repeat = int(env["action_repeat"])
        self.episode_seconds = float(env["episode_seconds"])
        self.max_steps = int(ceil(self.episode_seconds / (self.action_repeat * self.core.sim_dt)))
        self.residual_scale = np.asarray(config["residual_action_scale"], dtype=np.float32)
        if self.residual_scale.shape != (6,):
            raise ValueError("v040 residual_action_scale 必须是 6 维")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.tracker = BallEventTracker()
        self.step_count = 0
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.initial_ball = np.zeros(2, dtype=np.float64)
        self.receive_target = np.zeros(2, dtype=np.float64)
        self.pass_direction = np.asarray([1.0, 0.0], dtype=np.float64)
        self.receiver_control_steps = 0
        self.maximum_receiver_control_steps = 0
        self.transfer_ball_position: np.ndarray | None = None
        self.minimum_upright = 1.0
        self.minimum_height = 1.0
        self.success = False
        self.failed = False

    @property
    def passer_id(self) -> ContactIdentity:
        return ContactIdentity.from_agent_id(self.passer.agent_id)  # type: ignore[return-value]

    @property
    def receiver_id(self) -> ContactIdentity:
        return ContactIdentity.from_agent_id(self.receiver.agent_id)  # type: ignore[return-value]

    def _agent_features(self, agent: object, teammate: object, role: float, previous: np.ndarray) -> np.ndarray:
        rotation = np.asarray(agent.xmat, dtype=np.float64).reshape(3, 3)
        body_velocity = rotation.T @ np.asarray(agent._qvel[:3], dtype=np.float64)
        ball_relative = world_to_body(
            agent, np.asarray(self.core.simulation.ball.xpos[:2]) - np.asarray(agent.xpos[:2])
        )
        ball_velocity = world_to_body(agent, np.asarray(self.core.simulation.ball._qvel[:2]))
        teammate_relative = world_to_body(
            agent, np.asarray(teammate.xpos[:2]) - np.asarray(agent.xpos[:2])
        )
        target_relative = world_to_body(
            agent, self.receive_target - np.asarray(agent.xpos[:2], dtype=np.float64)
        )
        return np.asarray(
            [
                torso_upright(agent),
                float(agent.xpos[2]),
                np.clip(body_velocity[0] / 0.45, -3.0, 3.0),
                np.clip(body_velocity[1] / 0.24, -3.0, 3.0),
                np.clip(agent._qvel[5] / 2.0, -2.0, 2.0),
                np.clip(ball_relative[0] / 5.0, -2.0, 2.0),
                np.clip(ball_relative[1] / 3.0, -2.0, 2.0),
                np.clip(ball_velocity[0] / 3.0, -2.0, 2.0),
                np.clip(ball_velocity[1] / 3.0, -2.0, 2.0),
                np.clip(teammate_relative[0] / 8.0, -2.0, 2.0),
                np.clip(teammate_relative[1] / 5.0, -2.0, 2.0),
                np.clip(target_relative[0] / 8.0, -2.0, 2.0),
                np.clip(target_relative[1] / 5.0, -2.0, 2.0),
                role,
                *previous.tolist(),
            ],
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        observation = np.concatenate(
            [
                self._agent_features(self.passer, self.receiver, 1.0, self.previous_action[:3]),
                self._agent_features(self.receiver, self.passer, -1.0, self.previous_action[3:]),
            ]
        ).astype(np.float32)
        if observation.shape != (OBSERVATION_DIM,) or not np.isfinite(observation).all():
            raise FloatingPointError("v040 观测必须保持有限且为 34 维")
        return observation

    def teacher_action(self) -> np.ndarray:
        ball = np.asarray(self.core.simulation.ball.xpos[:2], dtype=np.float64)
        ball_progress = float(np.dot(ball - self.initial_ball, self.pass_direction))
        lateral_axis = np.asarray([-self.pass_direction[1], self.pass_direction[0]], dtype=np.float64)
        lateral_error = float(np.dot(ball - self.initial_ball, lateral_axis))
        direction = self.pass_direction - 1.25 * lateral_error * lateral_axis
        direction /= max(float(np.linalg.norm(direction)), 1.0e-6)
        behind = ball - 0.48 * direction
        passer_position = np.asarray(self.passer.xpos[:2], dtype=np.float64)
        if self.tracker.touched(self.passer_id) and ball_progress >= 1.45:
            passer_action = np.zeros(3, dtype=np.float32)
        elif float(np.linalg.norm(passer_position - behind)) > 0.22:
            passer_action = go_to_world(self.passer, behind, speed=0.34)
        else:
            passer_action = go_to_world(self.passer, ball + 1.40 * direction, speed=0.45)

        receiver_position = np.asarray(self.receiver.xpos[:2], dtype=np.float64)
        receiver_ball_distance = float(np.linalg.norm(receiver_position - ball))
        # 等球真正离开传球者一段距离后再接近，避免接球者过早把短碰撞算作传球。
        if receiver_ball_distance < 1.50 or ball_progress > 1.15:
            receiver_action = go_to_world(self.receiver, ball, speed=0.40)
        else:
            receiver_action = np.zeros(3, dtype=np.float32)
        return np.concatenate([passer_action, receiver_action]).astype(np.float32)

    def _metrics(self) -> dict[str, Any]:
        ball = np.asarray(self.core.simulation.ball.xpos[:2], dtype=np.float64)
        measured_ball = ball if self.transfer_ball_position is None else self.transfer_ball_position
        travel = float(np.dot(measured_ball - self.initial_ball, self.pass_direction))
        delta = measured_ball - self.initial_ball
        lateral = float(abs(self.pass_direction[0] * delta[1] - self.pass_direction[1] * delta[0]))
        safety = self.core.safety()
        ever_fallen = bool(self.minimum_height < 0.34 or self.minimum_upright < 0.20)
        return {
            "success": bool(self.success),
            "failed": bool(self.failed),
            "passer_touched": bool(self.tracker.touched(self.passer_id)),
            "receiver_touched": bool(self.tracker.touched(self.receiver_id)),
            "contact_transfer": bool(self.tracker.transferred(self.passer_id, self.receiver_id)),
            "ball_travel_m": travel,
            "ball_lateral_error_m": lateral,
            "receiver_control_seconds": float(
                self.maximum_receiver_control_steps * self.core.sim_dt
            ),
            "current_receiver_control_seconds": float(
                self.receiver_control_steps * self.core.sim_dt
            ),
            "current_receiver_lower_leg_contact": bool(
                self.tracker.has_lower_leg_contact(self.receiver_id)
            ),
            "current_ball_speed_mps": float(
                np.linalg.norm(np.asarray(self.core.simulation.ball._qvel[:2], dtype=np.float64))
            ),
            "episode_steps": int(self.step_count),
            "minimum_upright": float(self.minimum_upright),
            "minimum_height_m": float(self.minimum_height),
            "fell": ever_fallen,
            "finite_state": bool(
                safety["finite_qpos"] and safety["finite_qvel"] and safety["finite_ctrl"]
            ),
            "all_23_targets_sent": bool(safety["all_23_targets_sent"]),
            **{f"ball_events_{key}": value for key, value in self.tracker.summary().items()},
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
        lateral = float(self.rng.uniform(-0.65, 0.65))
        # 第一关先学习可验收的直线传接；斜传必须在本关通过后再扩展。
        receive_y = lateral
        passer_x = -4.4
        ball_x = -3.25
        receiver_x = float(self.rng.uniform(0.2, 0.8))
        self.core.reset_players(
            [(passer_x, lateral, 0.0), (receiver_x, receive_y, pi)],
            self.prepare_seconds,
        )
        self.core.simulation.ball.drop_at(ball_x, lateral)
        mujoco.mj_forward(self.core.simulation.mj_model, self.core.simulation.mj_data)
        self.core.simulation.referee.play_on()
        self.initial_ball = np.asarray(self.core.simulation.ball.xpos[:2], dtype=np.float64).copy()
        self.receive_target = np.asarray([receiver_x - 0.35, receive_y], dtype=np.float64)
        self.pass_direction = self.receive_target - self.initial_ball
        self.pass_direction /= max(float(np.linalg.norm(self.pass_direction)), 1.0e-6)
        self.tracker.reset()
        self.step_count = 0
        self.previous_action.fill(0.0)
        self.receiver_control_steps = 0
        self.maximum_receiver_control_steps = 0
        self.transfer_ball_position = None
        self.minimum_upright = min(torso_upright(self.passer), torso_upright(self.receiver))
        self.minimum_height = min(float(self.passer.xpos[2]), float(self.receiver.xpos[2]))
        self.success = False
        self.failed = False
        return self._observation(), self._metrics()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        residual = np.asarray(action, dtype=np.float32)
        if residual.shape != (ACTION_DIM,) or not np.isfinite(residual).all():
            raise FloatingPointError("v040 动作必须是有限的 6 维数组")
        residual = np.clip(residual, -1.0, 1.0)
        teacher = self.teacher_action()
        executed = np.clip(teacher + self.residual_scale * residual, -1.0, 1.0)
        old_ball = np.asarray(self.core.simulation.ball.xpos[:2], dtype=np.float64).copy()
        lateral_axis = np.asarray([-self.pass_direction[1], self.pass_direction[0]], dtype=np.float64)
        old_lateral_error = abs(float(np.dot(old_ball - self.initial_ball, lateral_axis)))
        old_receiver_distance = float(
            np.linalg.norm(np.asarray(self.receiver.xpos[:2]) - old_ball)
        )
        gates = self.config["quality_gates"]
        for _ in range(self.action_repeat):
            self.core.physics_step([executed[:3], executed[3:]])
            events = self.tracker.update(
                self.core.simulation.ball, float(self.core.simulation.mj_data.time)
            )
            if self.transfer_ball_position is None and any(
                event.kind == "transfer"
                and event.previous_actor == self.passer_id
                and event.actor == self.receiver_id
                for event in events
            ):
                self.transfer_ball_position = np.asarray(
                    self.core.simulation.ball.xpos[:2], dtype=np.float64
                ).copy()

            # "Controlled for N seconds" must be continuous physical control, not
            # merely being the most recent toucher while standing within 0.9 m.
            live_ball = np.asarray(self.core.simulation.ball.xpos[:2], dtype=np.float64)
            live_distance = float(
                np.linalg.norm(np.asarray(self.receiver.xpos[:2], dtype=np.float64) - live_ball)
            )
            live_speed = float(
                np.linalg.norm(np.asarray(self.core.simulation.ball._qvel[:2], dtype=np.float64))
            )
            controlled_now = bool(
                self.tracker.has_lower_leg_contact(self.receiver_id)
                and live_distance
                <= float(gates["maximum_receiver_control_distance_m"])
                and live_speed <= float(gates["maximum_receiver_control_ball_speed_mps"])
            )
            self.receiver_control_steps = self.receiver_control_steps + 1 if controlled_now else 0
            self.maximum_receiver_control_steps = max(
                self.maximum_receiver_control_steps, self.receiver_control_steps
            )

        self.step_count += 1
        self.previous_action[:] = residual
        ball = np.asarray(self.core.simulation.ball.xpos[:2], dtype=np.float64)
        new_lateral_error = abs(float(np.dot(ball - self.initial_ball, lateral_axis)))
        new_receiver_distance = float(np.linalg.norm(np.asarray(self.receiver.xpos[:2]) - ball))
        uprights = [torso_upright(self.passer), torso_upright(self.receiver)]
        heights = [float(self.passer.xpos[2]), float(self.receiver.xpos[2])]
        self.minimum_upright = min(self.minimum_upright, *uprights)
        self.minimum_height = min(self.minimum_height, *heights)
        metrics = self._metrics()
        self.success = bool(
            metrics["contact_transfer"]
            and metrics["ball_travel_m"] >= float(gates["minimum_ball_travel_m"])
            and metrics["ball_lateral_error_m"] <= float(gates["maximum_lateral_error_m"])
            and metrics["receiver_control_seconds"] >= float(gates["minimum_receiver_control_seconds"])
            and self.tracker.upper_body_contact_count == 0
            and not bool(metrics["fell"])
        )
        self.failed = bool(metrics["fell"] or self.tracker.upper_body_contact_count > 0)

        ball_progress = float(np.dot(ball - old_ball, self.pass_direction))
        receiver_approach = float(np.clip(old_receiver_distance - new_receiver_distance, -0.2, 0.2))
        reward = 8.0 * ball_progress + 0.8 * receiver_approach
        reward += 4.0 * float(np.clip(old_lateral_error - new_lateral_error, -0.15, 0.15))
        reward -= 0.01 * new_lateral_error
        reward += 0.05 if self.tracker.touched(self.passer_id) else 0.0
        reward += 0.10 if self.tracker.touched(self.receiver_id) else 0.0
        reward += 25.0 if self.success else 0.0
        reward -= 20.0 if self.failed else 0.0
        reward -= 0.005 * float(np.square(residual).sum())

        metrics = self._metrics()
        metrics.update(
            {
                "teacher_action": teacher.tolist(),
                "residual_action": residual.tolist(),
                "executed_action": executed.tolist(),
            }
        )
        terminated = bool(self.success or self.failed)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        return self._observation(), float(reward), terminated, truncated, metrics

    def close(self) -> None:
        return None
