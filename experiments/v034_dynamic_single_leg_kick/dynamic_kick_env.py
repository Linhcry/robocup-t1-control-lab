"""RCSSServerMJ environment for the v034 right-foot dynamic kick curriculum."""

from __future__ import annotations

from math import ceil, pi
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]

from locomotion.rcss_simulation import make_single_t1_simulation, torso_upright  # noqa: E402
from locomotion.rcss_t1_controller import PREPARE_QPOS, RcssT1Controller, read_joint_state  # noqa: E402
from locomotion.t1_spec import clamp_t1_targets  # noqa: E402


LEG_ACTION_INDEXES = np.arange(11, 23, dtype=np.int64)
LEG_ACTION_SCALE = np.asarray(
    [0.22, 0.10, 0.10, 0.28, 0.16, 0.08, 0.26, 0.12, 0.10, 0.32, 0.18, 0.09],
    dtype=np.float32,
)


def _smooth01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _key_pose(**updates: float) -> np.ndarray:
    pose = PREPARE_QPOS.copy()
    name_to_index = {
        "left_hip_pitch": 11,
        "left_hip_roll": 12,
        "left_hip_yaw": 13,
        "left_knee": 14,
        "left_ankle_pitch": 15,
        "left_ankle_roll": 16,
        "right_hip_pitch": 17,
        "right_hip_roll": 18,
        "right_hip_yaw": 19,
        "right_knee": 20,
        "right_ankle_pitch": 21,
        "right_ankle_roll": 22,
    }
    for name, value in updates.items():
        pose[name_to_index[name]] = float(value)
    return clamp_t1_targets(pose)


KICK_KEYFRAMES: tuple[tuple[float, np.ndarray], ...] = (
    (0.00, PREPARE_QPOS.copy()),
    (
        0.18,
        _key_pose(
            left_hip_roll=0.04,
            left_ankle_roll=-0.03,
            right_hip_roll=0.02,
            right_knee=0.35,
            right_ankle_roll=-0.01,
        ),
    ),
    (
        0.42,
        _key_pose(
            left_hip_roll=0.05,
            left_ankle_roll=-0.04,
            right_hip_pitch=-0.58,
            right_hip_roll=0.03,
            right_hip_yaw=-0.05,
            right_knee=1.10,
            right_ankle_pitch=-0.30,
            right_ankle_roll=-0.02,
        ),
    ),
    (
        0.60,
        _key_pose(
            left_hip_roll=0.05,
            left_ankle_roll=-0.04,
            right_hip_pitch=-0.82,
            right_hip_roll=0.03,
            right_hip_yaw=-0.03,
            right_knee=0.08,
            right_ankle_pitch=0.08,
            right_ankle_roll=-0.02,
        ),
    ),
    (
        0.76,
        _key_pose(
            left_hip_roll=0.04,
            left_ankle_roll=-0.03,
            right_hip_pitch=-0.35,
            right_hip_roll=0.02,
            right_knee=0.82,
            right_ankle_pitch=-0.24,
            right_ankle_roll=-0.01,
        ),
    ),
    (1.00, PREPARE_QPOS.copy()),
)


def dynamic_kick_prior(phase: float) -> np.ndarray:
    """Return a range-safe one-shot kick prior; PPO learns bounded residual corrections."""

    phase = float(np.clip(phase, 0.0, 1.0))
    for (start_phase, start_pose), (end_phase, end_pose) in zip(
        KICK_KEYFRAMES[:-1], KICK_KEYFRAMES[1:], strict=True
    ):
        if phase <= end_phase:
            alpha = _smooth01((phase - start_phase) / max(1.0e-6, end_phase - start_phase))
            return clamp_t1_targets((1.0 - alpha) * start_pose + alpha * end_pose)
    return PREPARE_QPOS.copy()


class DynamicSingleLegKickEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train a one-touch right-foot kick on top of the validated T1 stand controller."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any], *, seed: int | None = None) -> None:
        super().__init__()
        self.config = config
        env_cfg = config["environment"]
        self.control_repeat = int(env_cfg["control_repeat"])
        self.kick_seconds = float(env_cfg["kick_seconds"])
        self.episode_seconds = float(env_cfg["episode_seconds"])
        self.warmup_seconds = float(env_cfg["warmup_seconds"])
        self.target_retention = float(env_cfg["target_retention"])
        self.rng = np.random.default_rng(seed)

        official_root = PROJECT_ROOT.parents[1] / "third_party" / "booster_gym"
        self.controller = RcssT1Controller(official_root / "deploy" / "models" / "T1.pt")
        self.sim, self.agent = make_single_t1_simulation()
        self.sim_dt = float(self.sim.timestep)
        self.decision_dt = self.sim_dt * self.control_repeat
        self.max_episode_steps = int(ceil(self.episode_seconds / self.decision_dt))

        prefix = self.agent.agent_id.prefix
        self.left_foot_geom_id = mujoco.mj_name2id(
            self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, prefix + "left_foot"
        )
        self.right_foot_geom_id = mujoco.mj_name2id(
            self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, prefix + "right_foot"
        )
        self.ball_geom_id = mujoco.mj_name2id(self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        if min(self.left_foot_geom_id, self.right_foot_geom_id, self.ball_geom_id) < 0:
            raise RuntimeError("Unable to resolve T1 foot or ball geometry")

        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(82,), dtype=np.float32)
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.last_targets = PREPARE_QPOS.copy()
        self.step_count = 0
        self.initial_ball_position = np.zeros(3, dtype=np.float64)
        self.desired_direction = np.asarray([1.0, 0.0], dtype=np.float64)
        self.prev_right_foot_ball_distance = 0.0
        self.prev_ball_forward = 0.0
        self.prev_left_foot_xy = np.zeros(2, dtype=np.float64)
        self.max_ball_speed = 0.0
        self.ball_velocity_at_peak = np.zeros(3, dtype=np.float64)
        self.min_height = 0.0
        self.min_upright = 0.0
        self.touch_count = 0
        self.left_touch_count = 0
        self.right_contact_active = False
        self.left_contact_active = False
        self.contact_step: int | None = None
        self.recovery_time_s: float | None = None
        self.stable_after_contact_steps = 0
        self.fallen = False
        self.invalid = False

    @classmethod
    def from_config(cls, config: dict[str, Any], *, seed: int | None = None) -> "DynamicSingleLegKickEnv":
        return cls(config, seed=seed)

    def _foot_position(self, geom_id: int) -> np.ndarray:
        return np.asarray(self.sim.mj_data.geom_xpos[geom_id], dtype=np.float64).copy()

    def _phase(self) -> float:
        return float(np.clip(self.step_count * self.decision_dt / self.kick_seconds, 0.0, 1.0))

    def _contacts(self) -> tuple[bool, bool]:
        left = False
        right = False
        for contact in self.sim.mj_data.contact[: self.sim.mj_data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            left = left or pair == {self.ball_geom_id, self.left_foot_geom_id}
            right = right or pair == {self.ball_geom_id, self.right_foot_geom_id}
        return left, right

    def _record_contact_edges(self) -> tuple[bool, bool]:
        left, right = self._contacts()
        new_left = left and not self.left_contact_active
        new_right = right and not self.right_contact_active
        if new_left:
            self.left_touch_count += 1
        if new_right:
            self.touch_count += 1
            if self.contact_step is None:
                self.contact_step = self.step_count
        self.left_contact_active = left
        self.right_contact_active = right
        return new_left, new_right

    def _direction_error_deg(self) -> float:
        velocity = self.ball_velocity_at_peak[:2]
        speed = float(np.linalg.norm(velocity))
        if speed < 1.0e-6:
            return 180.0
        cosine = float(np.clip(np.dot(velocity / speed, self.desired_direction), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def _observation(self) -> np.ndarray:
        q, dq = read_joint_state(self.agent)
        rotation_world_from_body = np.asarray(self.agent.xmat, dtype=np.float32).reshape(3, 3)
        gravity_body = rotation_world_from_body.T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        angular_velocity_body = rotation_world_from_body.T @ np.asarray(self.agent._qvel[3:6], dtype=np.float32)
        linear_velocity_body = rotation_world_from_body.T @ np.asarray(self.agent._qvel[:3], dtype=np.float32)
        ball_relative_body = rotation_world_from_body.T @ (
            np.asarray(self.sim.ball.xpos, dtype=np.float32) - np.asarray(self.agent.xpos, dtype=np.float32)
        )
        ball_velocity_body = rotation_world_from_body.T @ np.asarray(self.sim.ball._qvel[:3], dtype=np.float32)
        left_foot_relative = rotation_world_from_body.T @ (
            self._foot_position(self.left_foot_geom_id).astype(np.float32)
            - np.asarray(self.agent.xpos, dtype=np.float32)
        )
        right_foot_to_ball = rotation_world_from_body.T @ (
            np.asarray(self.sim.ball.xpos, dtype=np.float32)
            - self._foot_position(self.right_foot_geom_id).astype(np.float32)
        )
        phase_angle = 2.0 * pi * self._phase()
        phase_features = np.asarray(
            [np.sin(phase_angle), np.cos(phase_angle), float(self.contact_step is not None)], dtype=np.float32
        )
        obs = np.concatenate(
            [
                gravity_body,
                0.5 * angular_velocity_body,
                0.5 * linear_velocity_body,
                q - PREPARE_QPOS,
                0.1 * dq,
                2.0 * ball_relative_body,
                0.5 * ball_velocity_body,
                left_foot_relative,
                right_foot_to_ball,
                phase_features,
                self.prev_action,
            ]
        ).astype(np.float32)
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(f"v034 observation shape changed: {obs.shape}")
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10.0, 10.0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.controller.reset_robot(self.sim, self.agent)
        for _ in range(round(self.warmup_seconds / self.sim_dt)):
            self.controller.prepare_step(self.sim, self.agent)
            self.sim.step([])

        ball_cfg = self.config["ball_randomization"]
        ball_x = float(self.rng.uniform(*ball_cfg["x_m"]))
        ball_y = float(self.rng.uniform(*ball_cfg["y_m"]))
        target_angle = float(np.deg2rad(self.rng.uniform(*ball_cfg["target_angle_deg"])))
        self.desired_direction = np.asarray([np.cos(target_angle), np.sin(target_angle)], dtype=np.float64)
        self.sim.ball.drop_at(float(self.agent.xpos[0] + ball_x), float(self.agent.xpos[1] + ball_y))
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)

        self.step_count = 0
        self.prev_action.fill(0.0)
        self.last_targets[:] = PREPARE_QPOS
        self.initial_ball_position = np.asarray(self.sim.ball.xpos, dtype=np.float64).copy()
        right_foot = self._foot_position(self.right_foot_geom_id)
        self.prev_right_foot_ball_distance = float(
            np.linalg.norm(np.asarray(self.sim.ball.xpos, dtype=np.float64) - right_foot)
        )
        self.prev_ball_forward = 0.0
        self.prev_left_foot_xy = self._foot_position(self.left_foot_geom_id)[:2]
        self.max_ball_speed = 0.0
        self.ball_velocity_at_peak.fill(0.0)
        self.min_height = float(self.agent.xpos[2])
        self.min_upright = float(torso_upright(self.agent))
        self.touch_count = 0
        self.left_touch_count = 0
        self.right_contact_active = False
        self.left_contact_active = False
        self.contact_step = None
        self.recovery_time_s = None
        self.stable_after_contact_steps = 0
        self.fallen = False
        self.invalid = False
        return self._observation(), self._episode_metrics()

    def _episode_metrics(self) -> dict[str, float | int | bool | None]:
        ball_delta = np.asarray(self.sim.ball.xpos, dtype=np.float64) - self.initial_ball_position
        forward_displacement = float(np.dot(ball_delta[:2], self.desired_direction))
        lateral_displacement = float(
            np.dot(ball_delta[:2], np.asarray([-self.desired_direction[1], self.desired_direction[0]]))
        )
        gates = self.config["episode_success"]
        direction_error = self._direction_error_deg()
        recovered = self.recovery_time_s is not None
        success = bool(
            self.touch_count == 1
            and self.left_touch_count == 0
            and self.max_ball_speed >= float(gates["ball_speed_mps"])
            and direction_error <= float(gates["direction_error_deg"])
            and forward_displacement >= float(gates["ball_forward_m"])
            and recovered
            and not self.fallen
            and not self.invalid
        )
        return {
            "phase": self._phase(),
            "touch_count": self.touch_count,
            "left_touch_count": self.left_touch_count,
            "contacted_ball": self.contact_step is not None,
            "max_ball_speed_mps": self.max_ball_speed,
            "direction_error_deg": direction_error,
            "ball_forward_m": forward_displacement,
            "ball_lateral_m": lateral_displacement,
            "min_robot_height_m": self.min_height,
            "min_robot_upright": self.min_upright,
            "recovery_time_s": self.recovery_time_s,
            "fallen": self.fallen,
            "invalid": self.invalid,
            "success": success,
            "all_23_targets_sent": self.controller.all_23_targets_sent,
        }

    def step(self, action: np.ndarray):
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (12,):
            raise ValueError(f"v034 action must have shape (12,), got {values.shape}")
        if not np.isfinite(values).all():
            self.invalid = True
            info = self._episode_metrics()
            return self._observation(), -50.0, True, False, info
        values = np.clip(values, -1.0, 1.0).astype(np.float32)

        phase = self._phase()
        prior = dynamic_kick_prior(phase)
        raw_targets = prior.copy()
        raw_targets[LEG_ACTION_INDEXES] += LEG_ACTION_SCALE * values
        safe_targets = clamp_t1_targets(raw_targets)
        clip_amount = float(np.mean(np.abs(raw_targets - safe_targets)))
        previous_peak_speed = self.max_ball_speed
        previous_touch_count = self.touch_count
        previous_left_touch_count = self.left_touch_count

        for _ in range(self.control_repeat):
            self.last_targets = self.controller.target_step(
                self.sim, self.agent, safe_targets, retention=self.target_retention
            )
            self.sim.step([])
            self._record_contact_edges()
            ball_velocity = np.asarray(self.sim.ball._qvel[:3], dtype=np.float64)
            ball_speed = float(np.linalg.norm(ball_velocity))
            if ball_speed > self.max_ball_speed:
                self.max_ball_speed = ball_speed
                self.ball_velocity_at_peak = ball_velocity.copy()

        self.step_count += 1
        self.prev_action, old_action = values.copy(), self.prev_action.copy()
        height = float(self.agent.xpos[2])
        upright = float(torso_upright(self.agent))
        self.min_height = min(self.min_height, height)
        self.min_upright = min(self.min_upright, upright)
        finite_state = bool(
            np.isfinite(self.agent.xpos).all()
            and np.isfinite(self.sim.ball.xpos).all()
            and np.isfinite(self.sim.mj_data.ctrl).all()
        )
        self.invalid = self.invalid or not finite_state
        self.fallen = self.fallen or height < 0.34 or upright < 0.20

        left_foot = self._foot_position(self.left_foot_geom_id)
        right_foot = self._foot_position(self.right_foot_geom_id)
        foot_ball_distance = float(np.linalg.norm(np.asarray(self.sim.ball.xpos) - right_foot))
        foot_progress = self.prev_right_foot_ball_distance - foot_ball_distance
        self.prev_right_foot_ball_distance = foot_ball_distance
        support_slip_speed = float(np.linalg.norm(left_foot[:2] - self.prev_left_foot_xy) / self.decision_dt)
        self.prev_left_foot_xy = left_foot[:2].copy()

        ball_delta = np.asarray(self.sim.ball.xpos, dtype=np.float64) - self.initial_ball_position
        ball_forward = float(np.dot(ball_delta[:2], self.desired_direction))
        ball_forward_progress = ball_forward - self.prev_ball_forward
        self.prev_ball_forward = ball_forward
        upright_score = float(np.clip((upright - 0.2) / 0.8, 0.0, 1.0))
        height_score = float(np.exp(-((height - 0.64) / 0.16) ** 2))
        stability_score = upright_score * height_score

        if self.contact_step is not None and self.recovery_time_s is None:
            root_angular_speed = float(np.linalg.norm(np.asarray(self.agent._qvel[3:6], dtype=np.float64)))
            stable_now = (
                self._phase() >= 0.78
                and height > 0.54
                and upright > 0.82
                and root_angular_speed < 1.6
            )
            self.stable_after_contact_steps = self.stable_after_contact_steps + 1 if stable_now else 0
            if self.stable_after_contact_steps >= 5:
                self.recovery_time_s = max(
                    0.0, (self.step_count - self.contact_step - self.stable_after_contact_steps + 1) * self.decision_dt
                )

        reward = 0.12 + 0.80 * stability_score
        reward += 4.0 * float(np.clip(foot_progress, -0.05, 0.05))
        reward += 0.12 * float(np.exp(-((foot_ball_distance / 0.18) ** 2)))
        reward += 8.0 * max(0.0, ball_forward_progress)
        reward += 4.0 * max(0.0, self.max_ball_speed - previous_peak_speed)
        reward -= 0.04 * min(support_slip_speed, 3.0)
        reward -= 0.012 * float(np.mean(np.square(values)))
        reward -= 0.025 * float(np.mean(np.square(values - old_action)))
        reward -= 0.8 * clip_amount
        if self.touch_count > previous_touch_count:
            reward += 6.0 if self.touch_count == 1 else -8.0
        if self.left_touch_count > previous_left_touch_count:
            reward -= 6.0
        if self.contact_step is not None:
            reward += 0.65 * stability_score

        terminated = bool(self.fallen or self.invalid)
        truncated = self.step_count >= self.max_episode_steps
        info = self._episode_metrics()
        if terminated:
            reward -= 40.0 if self.fallen else 50.0
        if (terminated or truncated) and bool(info["success"]):
            reward += 12.0
        return self._observation(), float(reward), terminated, truncated, info

    def close(self) -> None:
        return None
