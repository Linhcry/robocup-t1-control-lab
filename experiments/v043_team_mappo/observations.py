"""Masked local observations and privileged centralized critic state for v043."""

from __future__ import annotations

from math import atan2, pi
from collections.abc import Mapping

import numpy as np

from experiments.v043_team_mappo.ball_events import BallEventSnapshot
from experiments.v043_team_mappo.roles import home_position
from experiments.v043_team_mappo.spec import (
    CENTRAL_STATE_DIM,
    ENTITY_MASK_DIM,
    LOCAL_OBSERVATION_DIM,
    MAX_PLAYERS_PER_TEAM,
    ROLE_COUNT,
    Role,
)
from locomotion.rcss_simulation import torso_upright


VISION_RANGE_M = 20.0
VISION_HALF_ANGLE_RAD = pi / 3.0


def player_yaw(agent: object) -> float:
    rotation = np.asarray(agent.xmat, dtype=np.float64).reshape(3, 3)
    return float(atan2(rotation[1, 0], rotation[0, 0]))


def world_to_body(agent: object, vector_xy: np.ndarray) -> np.ndarray:
    yaw = player_yaw(agent)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    x_world, y_world = float(vector_xy[0]), float(vector_xy[1])
    return np.asarray(
        [cosine * x_world + sine * y_world, -sine * x_world + cosine * y_world],
        dtype=np.float32,
    )


def _entity(relative_body: np.ndarray, half_x: float, half_y: float) -> np.ndarray:
    distance = float(np.linalg.norm(relative_body))
    angle = abs(float(atan2(relative_body[1], relative_body[0])))
    visible = distance <= VISION_RANGE_M and angle <= VISION_HALF_ANGLE_RAD
    if not visible:
        return np.zeros(4, dtype=np.float32)
    return np.asarray(
        [
            np.clip(relative_body[0] / half_x, -1.0, 1.0),
            np.clip(relative_body[1] / half_y, -1.0, 1.0),
            np.clip(distance / VISION_RANGE_M, 0.0, 1.0),
            1.0,
        ],
        dtype=np.float32,
    )


def build_local_observation(
    simulation: object,
    agent: object,
    teammates: Mapping[int, object],
    opponents: Mapping[int, object],
    roles: Mapping[int, Role],
    event: BallEventSnapshot,
    *,
    side_sign: int,
    role_changed: bool,
) -> np.ndarray:
    """Build a fixed 92D tensor; absent or invisible entities have mask zero."""

    obs = np.zeros(LOCAL_OBSERVATION_DIM, dtype=np.float32)
    half_x = float(simulation.field.field_dim[0]) / 2.0
    half_y = float(simulation.field.field_dim[1]) / 2.0
    player_no = int(agent.agent_id.player_no)
    rotation = np.asarray(agent.xmat, dtype=np.float32).reshape(3, 3)
    body_velocity = rotation.T @ np.asarray(agent._qvel[:3], dtype=np.float32)
    obs[0:5] = (
        float(torso_upright(agent)),
        float(np.clip(agent.xpos[2] / 1.2, 0.0, 1.5)),
        float(np.clip(body_velocity[0] / 0.60, -2.0, 2.0)),
        float(np.clip(body_velocity[1] / 0.30, -2.0, 2.0)),
        float(np.clip(agent._qvel[5] / 2.0, -1.0, 1.0)),
    )

    ball_world_delta = np.asarray(simulation.ball.xpos[:2]) - np.asarray(agent.xpos[:2])
    ball_body = world_to_body(agent, ball_world_delta)
    ball_entity = _entity(ball_body, half_x, half_y)
    if ball_entity[3] > 0.5:
        ball_velocity_body = rotation.T @ np.asarray(simulation.ball._qvel[:3], dtype=np.float32)
        obs[5:10] = (
            ball_entity[0],
            ball_entity[1],
            float(np.clip(ball_velocity_body[0] / 3.0, -1.0, 1.0)),
            float(np.clip(ball_velocity_body[1] / 3.0, -1.0, 1.0)),
            1.0,
        )

    goal = np.asarray([side_sign * half_x, 0.0], dtype=np.float64)
    home = np.asarray(home_position(player_no, len(teammates), side_sign), dtype=np.float64)
    goal_body = world_to_body(agent, goal - np.asarray(agent.xpos[:2]))
    home_body = world_to_body(agent, home - np.asarray(agent.xpos[:2]))
    obs[10:14] = (
        np.clip(goal_body[0] / half_x, -2.0, 2.0),
        np.clip(goal_body[1] / half_y, -2.0, 2.0),
        np.clip(home_body[0] / half_x, -2.0, 2.0),
        np.clip(home_body[1] / half_y, -2.0, 2.0),
    )

    cursor = 14
    for teammate_no in range(1, MAX_PLAYERS_PER_TEAM + 1):
        if teammate_no == player_no:
            continue
        if teammate_no in teammates:
            relative = world_to_body(
                agent,
                np.asarray(teammates[teammate_no].xpos[:2]) - np.asarray(agent.xpos[:2]),
            )
            obs[cursor : cursor + 4] = _entity(relative, half_x, half_y)
        cursor += 4
    for opponent_no in range(1, MAX_PLAYERS_PER_TEAM + 1):
        if opponent_no in opponents:
            relative = world_to_body(
                agent,
                np.asarray(opponents[opponent_no].xpos[:2]) - np.asarray(agent.xpos[:2]),
            )
            obs[cursor : cursor + 4] = _entity(relative, half_x, half_y)
        cursor += 4

    obs[66 + int(roles[player_no])] = 1.0
    obs[70 + player_no - 1] = 1.0
    ball_visible = bool(obs[9] > 0.5)
    obs[77:84] = event.encoded_for_local_actor(
        side_sign,
        player_no,
        ball_visible=ball_visible,
    )
    state = simulation.game_state
    own_score = state.left_team_score if side_sign > 0 else state.right_team_score
    opponent_score = state.right_team_score if side_sign > 0 else state.left_team_score
    play_modes = list(type(state.play_mode))
    play_mode_index = play_modes.index(state.play_mode)
    obs[84:92] = (
        float(side_sign),
        float(np.clip(state.play_time / 600.0, 0.0, 1.0)),
        float(np.clip((own_score - opponent_score) / 3.0, -1.0, 1.0)),
        float(play_mode_index) / max(1, len(play_modes) - 1),
        float(len(teammates)) / MAX_PLAYERS_PER_TEAM,
        float(ball_visible),
        float(roles[player_no] == Role.PRESSER),
        float(role_changed),
    )
    if obs.shape != (LOCAL_OBSERVATION_DIM,) or not np.isfinite(obs).all():
        raise FloatingPointError("v043 local observation must be finite and 92-dimensional")
    return obs


def entity_mask(observation: np.ndarray) -> np.ndarray:
    """Extract ball + six teammate + seven opponent visibility masks."""

    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (LOCAL_OBSERVATION_DIM,):
        raise ValueError(f"expected ({LOCAL_OBSERVATION_DIM},), got {obs.shape}")
    masks = [float(obs[9])]
    masks.extend(float(obs[14 + slot * 4 + 3]) for slot in range(6))
    masks.extend(float(obs[38 + slot * 4 + 3]) for slot in range(7))
    result = np.asarray(masks, dtype=np.float32)
    if result.shape != (ENTITY_MASK_DIM,):
        raise AssertionError(result.shape)
    return result


def build_central_state(
    simulation: object,
    own_players: Mapping[int, object],
    opponents: Mapping[int, object],
    roles: Mapping[int, Role],
    event: BallEventSnapshot,
    *,
    side_sign: int,
    possession_side: int,
) -> np.ndarray:
    """Build a 96D privileged team state used only by the critic during training."""

    state = np.zeros(CENTRAL_STATE_DIM, dtype=np.float32)
    half_x = float(simulation.field.field_dim[0]) / 2.0
    half_y = float(simulation.field.field_dim[1]) / 2.0
    side = 1 if side_sign > 0 else -1
    state[0:4] = (
        np.clip(side * simulation.ball.xpos[0] / half_x, -1.2, 1.2),
        np.clip(simulation.ball.xpos[1] / half_y, -1.2, 1.2),
        np.clip(side * simulation.ball._qvel[0] / 3.0, -1.0, 1.0),
        np.clip(simulation.ball._qvel[1] / 3.0, -1.0, 1.0),
    )
    cursor = 4
    for collection in (own_players, opponents):
        for player_no in range(1, MAX_PLAYERS_PER_TEAM + 1):
            if player_no in collection:
                agent = collection[player_no]
                state[cursor : cursor + 4] = (
                    np.clip(side * agent.xpos[0] / half_x, -1.2, 1.2),
                    np.clip(agent.xpos[1] / half_y, -1.2, 1.2),
                    np.clip(torso_upright(agent), -1.0, 1.0),
                    1.0,
                )
            cursor += 4
    for player_no in range(1, MAX_PLAYERS_PER_TEAM + 1):
        if player_no in roles:
            state[cursor + int(roles[player_no])] = 1.0
        cursor += ROLE_COUNT
    game = simulation.game_state
    own_score = game.left_team_score if side > 0 else game.right_team_score
    opponent_score = game.right_team_score if side > 0 else game.left_team_score
    play_modes = list(type(game.play_mode))
    state[88:96] = (
        float(side),
        float(np.clip(game.play_time / 600.0, 0.0, 1.0)),
        float(np.clip((own_score - opponent_score) / 3.0, -1.0, 1.0)),
        float(play_modes.index(game.play_mode)) / max(1, len(play_modes) - 1),
        float(len(own_players)) / MAX_PLAYERS_PER_TEAM,
        float(possession_side == side),
        float(possession_side == -side),
        float(event.turnover),
    )
    if state.shape != (CENTRAL_STATE_DIM,) or not np.isfinite(state).all():
        raise FloatingPointError("v043 central state must be finite and 96-dimensional")
    return state
