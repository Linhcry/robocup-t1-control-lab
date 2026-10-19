"""RCSSServerMJ adapter for the official 23D mjlab getup actor."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from locomotion.rcss_t1_controller import read_joint_state
from locomotion.t1_spec import (
    T1_ACTION_DELTA_LIMIT_RAD,
    T1_MOTOR_NAMES,
    clamp_t1_targets,
    filter_t1_dynamic_targets,
)

from .mjlab_getup_policy import MjlabGetupPolicy


# Gains and source-policy effort limits from the official mjlab Booster T1
# constants.  The RCSS competition model has stricter joint-level mechanical
# caps for several joints; deployment must use those effective caps below.
GETUP_KP = np.asarray(
    [
        1.7765288, 1.7765288,
        27.884396, 27.884396, 27.884396, 27.884396,
        27.884396, 27.884396, 27.884396, 27.884396,
        47.189045,
        51.70765, 47.189045, 47.189045, 62.77187, 33.51244, 33.51244,
        51.70765, 47.189045, 47.189045, 62.77187, 33.51244, 33.51244,
    ],
    dtype=np.float32,
)
GETUP_KD = np.asarray(
    [
        0.22619468, 0.22619468,
        3.5503516, 3.5503516, 3.5503516, 3.5503516,
        3.5503516, 3.5503516, 3.5503516, 3.5503516,
        6.008296,
        6.583622, 6.008296, 6.008296, 7.9923625, 4.2669363, 4.2669363,
        6.583622, 6.008296, 6.008296, 7.9923625, 4.2669363, 4.2669363,
    ],
    dtype=np.float32,
)
GETUP_EFFORT_LIMIT = np.asarray(
    [
        7.0, 7.0,
        36.0, 36.0, 36.0, 36.0,
        36.0, 36.0, 36.0, 36.0,
        40.0,
        55.0, 40.0, 40.0, 65.0, 50.0, 50.0,
        55.0, 40.0, 40.0, 65.0, 50.0, 50.0,
    ],
    dtype=np.float32,
)

# Effective joint torque caps enforced by the installed RCSS T1 robot.xml/MJB.
# These are lower than (or equal to) the source-policy actuator caps.  Keeping
# this explicit lets training, deployment and the safety recorder use the same
# 23-joint contract instead of under-reporting saturation against the larger
# source limits.
RCSS_EFFECTIVE_EFFORT_LIMIT = np.asarray(
    [
        7.0, 7.0,
        18.0, 18.0, 18.0, 18.0,
        18.0, 18.0, 18.0, 18.0,
        30.0,
        45.0, 30.0, 30.0, 60.0, 20.0, 15.0,
        45.0, 30.0, 30.0, 60.0, 20.0, 15.0,
    ],
    dtype=np.float32,
)


class RcssMjlabGetupController:
    """Run 50 Hz actor inference and safe torque-limited RCSS control."""

    action_scale = 0.6

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        effort_limits: np.ndarray = RCSS_EFFECTIVE_EFFORT_LIMIT,
        max_target_delta_rad: float = T1_ACTION_DELTA_LIMIT_RAD,
    ) -> None:
        self.policy = MjlabGetupPolicy(checkpoint, device=device)
        limits = np.asarray(effort_limits, dtype=np.float32)
        if limits.shape != (23,) or not np.isfinite(limits).all():
            raise ValueError("getup effort limits must be one finite 23D vector")
        if np.any(limits <= 0.0):
            raise ValueError("getup effort limits must all be positive")
        if np.any(limits > RCSS_EFFECTIVE_EFFORT_LIMIT + 1e-6):
            raise ValueError("getup effort limits exceed the RCSS mechanical contract")
        if not np.isfinite(max_target_delta_rad) or max_target_delta_rad <= 0.0:
            raise ValueError("max_target_delta_rad must be one positive finite scalar")
        self.effort_limits = limits.copy()
        self.max_target_delta_rad = float(max_target_delta_rad)
        self.target = np.zeros(23, dtype=np.float32)
        self.last_requested_torque = np.zeros(23, dtype=np.float32)
        self.last_sent_torque = np.zeros(23, dtype=np.float32)

    def reset(self, joint_pos: np.ndarray) -> None:
        self.policy.reset()
        self.target[:] = clamp_t1_targets(np.asarray(joint_pos, dtype=np.float32))
        self.last_requested_torque.fill(0.0)
        self.last_sent_torque.fill(0.0)

    def policy_step(self, simulation: object, agent: object) -> np.ndarray:
        joint_pos, joint_vel = read_joint_state(agent)
        rotation = np.asarray(agent.xmat, dtype=np.float32).reshape(3, 3)
        projected_gravity = rotation.T @ np.asarray(
            [0.0, 0.0, -1.0], dtype=np.float32
        )
        sensor = simulation.mj_data.sensor(agent.agent_id.prefix + "torso_gyro")
        if sensor is None:
            raise RuntimeError("RCSS torso gyro sensor is missing")
        body_ang_vel = np.asarray(sensor.data, dtype=np.float32).copy()
        action = self.policy.infer(
            base_ang_vel=body_ang_vel,
            projected_gravity=projected_gravity,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )
        requested = joint_pos + self.action_scale * action
        if requested.shape != (23,) or not np.isfinite(requested).all():
            raise FloatingPointError("invalid 23D target before RCSS clamp")
        self.target[:] = filter_t1_dynamic_targets(
            requested,
            joint_pos,
            joint_vel,
            max_delta_rad=self.max_target_delta_rad,
        )
        return action

    def hold_current(self, agent: object) -> None:
        joint_pos, _ = read_joint_state(agent)
        self.target[:] = clamp_t1_targets(joint_pos)

    def physics_step(self, simulation: object, agent: object) -> dict[str, float]:
        joint_pos, joint_vel = read_joint_state(agent)
        requested_torque = GETUP_KP * (self.target - joint_pos) - GETUP_KD * joint_vel
        if not np.isfinite(requested_torque).all():
            raise FloatingPointError("non-finite getup torque")
        sent_torque = np.clip(
            requested_torque, -self.effort_limits, self.effort_limits
        ).astype(np.float32)
        prefix = agent.agent_id.prefix
        for motor, current, torque in zip(
            T1_MOTOR_NAMES, joint_pos, sent_torque, strict=True
        ):
            simulation.ctrl_motor(
                prefix + motor,
                float(current),
                0.0,
                0.0,
                0.0,
                float(torque),
            )
        self.last_requested_torque[:] = requested_torque
        self.last_sent_torque[:] = sent_torque
        return {
            "saturation_fraction": float(
                np.mean(np.abs(requested_torque) >= self.effort_limits * 0.999)
            ),
            "maximum_requested_torque_nm": float(np.max(np.abs(requested_torque))),
            "maximum_sent_torque_nm": float(np.max(np.abs(sent_torque))),
        }
