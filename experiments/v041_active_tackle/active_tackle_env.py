"""一名带球队员对一名防守队员的主动抢断课程。"""

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


OBSERVATION_DIM = 19
ACTION_DIM = 3
RIGHT_GOAL = np.asarray([27.5, 0.0], dtype=np.float64)


class ActiveTackleEnv(gym.Env[np.ndarray, np.ndarray]):
    """攻击者由固定教师控球，学习方只输出防守者三维残差动作。"""

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
                AgentParams(team_name="Attack", player_no=2),
                AgentParams(team_name="Defense", player_no=2),
            ],
            **kwargs,
        )
        self.attacker, self.defender = self.core.agents
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.prepare_seconds = float(env["prepare_seconds"])
        self.action_repeat = int(env["action_repeat"])
        self.episode_seconds = float(env["episode_seconds"])
        self.max_steps = int(ceil(self.episode_seconds / (self.action_repeat * self.core.sim_dt)))
        self.residual_scale = np.asarray(config["residual_action_scale"], dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.tracker = BallEventTracker()
        self.step_count = 0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.turnover_ball_x: float | None = None
        self.maximum_clearance = 0.0
        self.control_steps = 0
        self.maximum_control_steps = 0
        self.minimum_upright = 1.0
        self.minimum_height = 1.0
        self.ever_fallen = False
        self.cross_team_contact_frames = 0
        self.upper_body_collision_frames = 0
        self.maximum_opponent_contact_force_n = 0.0
        self.foul_detected = False
        self.success = False
        self.failed = False

    @property
    def attacker_id(self) -> ContactIdentity:
        return ContactIdentity.from_agent_id(self.attacker.agent_id)  # type: ignore[return-value]

    @property
    def defender_id(self) -> ContactIdentity:
        return ContactIdentity.from_agent_id(self.defender.agent_id)  # type: ignore[return-value]

    def _observation(self) -> np.ndarray:
        agent = self.defender
        rotation = np.asarray(agent.xmat, dtype=np.float64).reshape(3, 3)
        body_velocity = rotation.T @ np.asarray(agent._qvel[:3], dtype=np.float64)
        ball_relative = world_to_body(
            agent, np.asarray(self.core.simulation.ball.xpos[:2]) - np.asarray(agent.xpos[:2])
        )
        ball_velocity = world_to_body(agent, np.asarray(self.core.simulation.ball._qvel[:2]))
        attacker_relative = world_to_body(
            agent, np.asarray(self.attacker.xpos[:2]) - np.asarray(agent.xpos[:2])
        )
        goal_relative = world_to_body(agent, RIGHT_GOAL - np.asarray(agent.xpos[:2]))
        observation = np.asarray(
            [
                torso_upright(agent),
                float(agent.xpos[2]),
                np.clip(body_velocity[0] / 0.45, -3.0, 3.0),
                np.clip(body_velocity[1] / 0.24, -3.0, 3.0),
                np.clip(agent._qvel[5] / 2.0, -2.0, 2.0),
                np.clip(ball_relative[0] / 4.0, -2.0, 2.0),
                np.clip(ball_relative[1] / 3.0, -2.0, 2.0),
                np.clip(ball_velocity[0] / 3.0, -2.0, 2.0),
                np.clip(ball_velocity[1] / 3.0, -2.0, 2.0),
                np.clip(attacker_relative[0] / 4.0, -2.0, 2.0),
                np.clip(attacker_relative[1] / 3.0, -2.0, 2.0),
                np.clip(goal_relative[0] / 27.5, -2.0, 2.0),
                np.clip(goal_relative[1] / 18.0, -2.0, 2.0),
                float(self.tracker.touched(self.attacker_id)),
                float(self.tracker.last_distinct_toucher == self.defender_id),
                np.clip(self.step_count / max(1, self.max_steps), 0.0, 1.0),
                *self.previous_action.tolist(),
            ],
            dtype=np.float32,
        )
        if observation.shape != (OBSERVATION_DIM,) or not np.isfinite(observation).all():
            raise FloatingPointError("v041 观测必须保持有限且为 19 维")
        return observation

    def _attacker_teacher(self) -> np.ndarray:
        if self.tracker.last_distinct_toucher == self.defender_id:
            return go_to_world(self.attacker, np.asarray(self.attacker.xpos[:2]), speed=0.20)
        ball = np.asarray(self.core.simulation.ball.xpos[:2], dtype=np.float64)
        direction = RIGHT_GOAL - ball
        direction /= max(float(np.linalg.norm(direction)), 1.0e-6)
        behind = ball - 0.43 * direction
        if float(np.linalg.norm(np.asarray(self.attacker.xpos[:2]) - behind)) > 0.21:
            return go_to_world(self.attacker, behind, speed=0.32)
        return go_to_world(self.attacker, ball + 1.1 * direction, speed=0.40)

    def defender_teacher(self) -> np.ndarray:
        ball = np.asarray(self.core.simulation.ball.xpos[:2], dtype=np.float64)
        if self.tracker.last_distinct_toucher == self.defender_id:
            if self.turnover_ball_x is not None and self.maximum_clearance >= 0.25:
                return np.zeros(3, dtype=np.float32)
            clear_direction = np.asarray([-1.0, 0.0], dtype=np.float64)
            behind = ball - 0.43 * clear_direction
            if float(np.linalg.norm(np.asarray(self.defender.xpos[:2]) - behind)) > 0.20:
                return go_to_world(self.defender, behind, speed=0.32)
            return go_to_world(self.defender, ball + 1.2 * clear_direction, speed=0.40)
        velocity = np.asarray(self.core.simulation.ball._qvel[:2], dtype=np.float64)
        predicted = ball + 0.30 * velocity
        goal_side = predicted + np.asarray([0.35, 0.0], dtype=np.float64)
        distance = float(np.linalg.norm(np.asarray(self.defender.xpos[:2]) - ball))
        target = ball if distance < 0.85 else goal_side
        return go_to_world(self.defender, target, speed=0.40)

    def _audit_cross_team_contacts(self) -> None:
        """Conservatively reject body-to-body charging in this tackle course."""

        model = self.core.simulation.mj_model
        data = self.core.simulation.mj_data
        attacker_prefix = str(self.attacker.agent_id.prefix)
        defender_prefix = str(self.defender.agent_id.prefix)
        upper_markers = (
            "head",
            "torso",
            "trunk",
            "chest",
            "waist",
            "shoulder",
            "upper_arm",
            "forearm",
            "hand",
        )
        frame_contact = False
        frame_upper_body = False
        wrench = np.zeros(6, dtype=np.float64)
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
            ) or ""
            geom2 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
            ) or ""
            cross_team = bool(
                (geom1.startswith(attacker_prefix) and geom2.startswith(defender_prefix))
                or (geom2.startswith(attacker_prefix) and geom1.startswith(defender_prefix))
            )
            if not cross_team:
                continue
            frame_contact = True
            frame_upper_body = frame_upper_body or any(
                marker in geom1.lower() or marker in geom2.lower()
                for marker in upper_markers
            )
            mujoco.mj_contactForce(model, data, index, wrench)
            self.maximum_opponent_contact_force_n = max(
                self.maximum_opponent_contact_force_n,
                float(np.linalg.norm(wrench[:3])),
            )
        if frame_contact:
            self.cross_team_contact_frames += 1
            # RCSS does not expose a stable per-step charging-foul flag here.  The
            # strict course therefore treats any body-to-body contact as illegal;
            # a valid tackle must win the ball without charging the opponent.
            self.foul_detected = True
        if frame_upper_body:
            self.upper_body_collision_frames += 1

    def _metrics(self) -> dict[str, Any]:
        ball_x = float(self.core.simulation.ball.xpos[0])
        current_clearance = 0.0 if self.turnover_ball_x is None else float(self.turnover_ball_x - ball_x)
        return {
            "success": bool(self.success),
            "failed": bool(self.failed),
            "attacker_touched": bool(self.tracker.touched(self.attacker_id)),
            "defender_touched": bool(self.tracker.touched(self.defender_id)),
            "turnover": bool(self.tracker.transferred(self.attacker_id, self.defender_id)),
            "clearance_away_from_own_goal_m": float(self.maximum_clearance),
            "current_clearance_m": current_clearance,
            "defender_control_seconds": float(
                self.maximum_control_steps * self.action_repeat * self.core.sim_dt
            ),
            "current_defender_control_seconds": float(
                self.control_steps * self.action_repeat * self.core.sim_dt
            ),
            "upper_body_contacts": int(self.tracker.upper_body_contact_count),
            "cross_team_contact_frames": int(self.cross_team_contact_frames),
            "upper_body_collision_frames": int(self.upper_body_collision_frames),
            "maximum_opponent_contact_force_n": float(
                self.maximum_opponent_contact_force_n
            ),
            "foul_detected": bool(self.foul_detected),
            "minimum_upright": float(self.minimum_upright),
            "minimum_height_m": float(self.minimum_height),
            "fell": bool(self.ever_fallen),
            "episode_steps": int(self.step_count),
            "all_23_targets_sent": bool(
                all(controller.all_23_targets_sent for controller in self.core.controllers.values())
            ),
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
        y = float(self.rng.uniform(-1.0, 1.0))
        defender_y = y + float(self.rng.uniform(-0.30, 0.30))
        self.core.reset_players(
            [(-2.0, y, 0.0), (1.15, defender_y, pi)],
            self.prepare_seconds,
        )
        self.core.simulation.ball.drop_at(-0.85, y)
        mujoco.mj_forward(self.core.simulation.mj_model, self.core.simulation.mj_data)
        self.core.simulation.referee.play_on()
        self.tracker.reset()
        self.step_count = 0
        self.previous_action.fill(0.0)
        self.turnover_ball_x = None
        self.maximum_clearance = 0.0
        self.control_steps = 0
        self.maximum_control_steps = 0
        self.minimum_upright = min(torso_upright(self.attacker), torso_upright(self.defender))
        self.minimum_height = min(float(self.attacker.xpos[2]), float(self.defender.xpos[2]))
        self.ever_fallen = False
        self.cross_team_contact_frames = 0
        self.upper_body_collision_frames = 0
        self.maximum_opponent_contact_force_n = 0.0
        self.foul_detected = False
        self.success = False
        self.failed = False
        return self._observation(), self._metrics()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        residual = np.asarray(action, dtype=np.float32)
        if residual.shape != (ACTION_DIM,) or not np.isfinite(residual).all():
            raise FloatingPointError("v041 动作必须是有限的三维数组")
        residual = np.clip(residual, -1.0, 1.0)
        defender_teacher = self.defender_teacher()
        defender_action = np.clip(
            defender_teacher + self.residual_scale * residual, -1.0, 1.0
        )
        old_defender_distance = float(
            np.linalg.norm(
                np.asarray(self.defender.xpos[:2])
                - np.asarray(self.core.simulation.ball.xpos[:2])
            )
        )
        old_ball_x = float(self.core.simulation.ball.xpos[0])
        for _ in range(self.action_repeat):
            self.core.physics_step([self._attacker_teacher(), defender_action])
            self._audit_cross_team_contacts()
            events = self.tracker.update(
                self.core.simulation.ball, float(self.core.simulation.mj_data.time)
            )
            if self.turnover_ball_x is None and any(
                event.kind == "transfer"
                and event.previous_actor == self.attacker_id
                and event.actor == self.defender_id
                for event in events
            ):
                self.turnover_ball_x = float(self.core.simulation.ball.xpos[0])

        self.step_count += 1
        self.previous_action[:] = residual
        ball = np.asarray(self.core.simulation.ball.xpos[:2], dtype=np.float64)
        if self.turnover_ball_x is not None:
            self.maximum_clearance = max(
                self.maximum_clearance, float(self.turnover_ball_x - float(ball[0]))
            )
        defender_distance = float(np.linalg.norm(np.asarray(self.defender.xpos[:2]) - ball))
        uprights = [torso_upright(self.attacker), torso_upright(self.defender)]
        heights = [float(self.attacker.xpos[2]), float(self.defender.xpos[2])]
        self.minimum_upright = min(self.minimum_upright, *uprights)
        self.minimum_height = min(self.minimum_height, *heights)
        self.ever_fallen = self.ever_fallen or any(
            height < 0.34 or upright < 0.20
            for height, upright in zip(heights, uprights, strict=True)
        )
        if self.tracker.last_distinct_toucher == self.defender_id and defender_distance <= 0.90:
            self.control_steps += 1
            self.maximum_control_steps = max(self.maximum_control_steps, self.control_steps)
        else:
            self.control_steps = 0

        gates = self.config["quality_gates"]
        metrics = self._metrics()
        self.success = bool(
            metrics["turnover"]
            and metrics["defender_control_seconds"] >= float(gates["minimum_control_seconds"])
            and metrics["clearance_away_from_own_goal_m"]
            >= float(gates["minimum_clearance_m"])
            and metrics["upper_body_contacts"] == 0
            and not metrics["foul_detected"]
            and not metrics["fell"]
        )
        self.failed = bool(
            metrics["fell"]
            or metrics["upper_body_contacts"] > 0
            or metrics["foul_detected"]
        )
        approach = float(np.clip(old_defender_distance - defender_distance, -0.2, 0.2))
        away_progress = float(np.clip(old_ball_x - float(ball[0]), -0.2, 0.2))
        reward = 1.5 * approach
        if self.turnover_ball_x is not None:
            reward += 8.0 * away_progress
        reward += 35.0 if self.success else 0.0
        reward -= 25.0 if self.failed else 0.0
        reward -= 0.005 * float(np.square(residual).sum())
        metrics = self._metrics()
        metrics.update(
            {
                "defender_teacher_action": defender_teacher.tolist(),
                "residual_action": residual.tolist(),
                "executed_defender_action": defender_action.tolist(),
            }
        )
        terminated = bool(self.success or self.failed)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        return self._observation(), float(reward), terminated, truncated, metrics

    def close(self) -> None:
        return None
