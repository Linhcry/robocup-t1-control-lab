"""Safe walking action adapter and privileged dynamic-role heuristic baseline."""

from __future__ import annotations

from math import atan2, pi
from collections.abc import Mapping

import numpy as np

from experiments.v043_team_mappo.observations import world_to_body
from experiments.v043_team_mappo.roles import home_position
from experiments.v043_team_mappo.spec import ACTION_DIM, Role
from locomotion.official_t1_policy import MotionCommand


MAX_VX_MPS = 0.45
MAX_VY_MPS = 0.24
MAX_YAW_RATE_RPS = 0.20


def _wrap_angle(angle: float) -> float:
    return float((angle + pi) % (2.0 * pi) - pi)


def _target_action(agent: object, target_xy: np.ndarray, *, speed_scale: float = 1.0) -> np.ndarray:
    relative_body = world_to_body(agent, target_xy - np.asarray(agent.xpos[:2], dtype=np.float64))
    distance = float(np.linalg.norm(relative_body))
    angle = float(atan2(relative_body[1], relative_body[0]))
    forward = np.clip(distance / 1.8, 0.0, 1.0) * np.clip(1.0 - abs(angle) / 1.5, 0.12, 1.0)
    lateral = np.clip(relative_body[1] / 2.0, -0.70, 0.70)
    yaw = np.clip(angle / 0.8, -1.0, 1.0)
    if distance < 0.25:
        forward = 0.0
        lateral = 0.0
    return np.asarray(
        [speed_scale * forward, speed_scale * lateral, yaw],
        dtype=np.float32,
    )


def privileged_role_heuristic_actions(
    simulation: object,
    players: Mapping[int, object],
    roles: Mapping[int, Role],
    *,
    side_sign: int,
) -> np.ndarray:
    """Use simulator ground truth to produce a curriculum teacher, not a learned policy."""

    side = 1 if side_sign > 0 else -1
    half_x = float(simulation.field.field_dim[0]) / 2.0
    half_y = float(simulation.field.field_dim[1]) / 2.0
    ball = np.asarray(simulation.ball.xpos[:2], dtype=np.float64)
    ball_velocity = np.asarray(simulation.ball._qvel[:2], dtype=np.float64)
    own_goal = np.asarray([-side * half_x, 0.0], dtype=np.float64)
    actions: list[np.ndarray] = []
    supporter_numbers = [number for number, role in roles.items() if role == Role.SUPPORTER]
    for player_no in range(1, len(players) + 1):
        agent = players[player_no]
        role = roles[player_no]
        if role == Role.GOALKEEPER:
            target = np.asarray(
                [
                    -side * (half_x - 1.0),
                    np.clip(0.38 * ball[1], -2.6, 2.6),
                ],
                dtype=np.float64,
            )
            if np.linalg.norm(ball - np.asarray(agent.xpos[:2])) < 3.0:
                target[0] = np.clip(ball[0], own_goal[0] - 0.5, own_goal[0] + side * 3.0)
            action = _target_action(agent, target, speed_scale=0.70)
        elif role == Role.PRESSER:
            predicted_ball = ball + np.clip(np.linalg.norm(ball_velocity), 0.0, 2.0) * 0.20 * ball_velocity
            behind = np.asarray([-side * 0.25, 0.0], dtype=np.float64)
            action = _target_action(agent, predicted_ball + behind, speed_scale=1.0)
        elif role == Role.SUPPORTER:
            support_rank = supporter_numbers.index(player_no) if player_no in supporter_numbers else 0
            lane = 1.0 if (player_no + support_rank) % 2 == 0 else -1.0
            target = ball + np.asarray([-side * (2.0 + 0.8 * support_rank), lane * (2.0 + support_rank)], dtype=np.float64)
            target[0] = np.clip(target[0], -half_x + 2.0, half_x - 2.0)
            target[1] = np.clip(target[1], -half_y + 1.5, half_y - 1.5)
            action = _target_action(agent, target, speed_scale=0.82)
        else:
            target = own_goal + 0.42 * (ball - own_goal)
            home = np.asarray(home_position(player_no, len(players), side), dtype=np.float64)
            target[1] = 0.55 * target[1] + 0.45 * home[1]
            action = _target_action(agent, target, speed_scale=0.78)
        actions.append(action)
    result = np.stack(actions).astype(np.float32)
    if result.shape != (len(players), ACTION_DIM) or not np.isfinite(result).all():
        raise FloatingPointError("privileged role heuristic produced invalid actions")
    return np.clip(result, -1.0, 1.0)


def action_to_motion_command(action: np.ndarray, observation: np.ndarray) -> MotionCommand:
    """Map a finite normalized team action into the validated walking envelope."""

    safe = np.asarray(action, dtype=np.float32)
    if safe.shape != (ACTION_DIM,) or not np.isfinite(safe).all():
        raise FloatingPointError(f"v043 action must be finite ({ACTION_DIM},)")
    safe = np.clip(safe, -1.0, 1.0)
    ball_angle = float(atan2(observation[6], observation[5])) if observation[9] > 0.5 else 0.0
    return MotionCommand(
        vx=float(MAX_VX_MPS * safe[0]),
        vy=float(MAX_VY_MPS * safe[1]),
        yaw_rate=float(MAX_YAW_RATE_RPS * safe[2]),
        head_yaw=float(np.clip(ball_angle, -0.8, 0.8)),
        head_pitch=0.28,
    )
