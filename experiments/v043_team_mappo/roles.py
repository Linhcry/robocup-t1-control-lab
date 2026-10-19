"""Deterministic dynamic-role teacher used to bootstrap recurrent team learning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import atan2, pi

import numpy as np

from experiments.v043_team_mappo.ball_events import (
    POSSESSION_HOLD_WINDOW_S,
    BallEventSnapshot,
)
from experiments.v043_team_mappo.spec import Role, validate_players_per_team


@dataclass(frozen=True)
class RoleAssignment:
    roles: dict[int, Role]
    possession_side: int
    changed_players: int


def _body_relative(observer: object, target_xy: np.ndarray) -> np.ndarray:
    rotation = np.asarray(observer.xmat, dtype=np.float64).reshape(3, 3)
    yaw = float(atan2(rotation[1, 0], rotation[0, 0]))
    delta = target_xy - np.asarray(observer.xpos[:2], dtype=np.float64)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1]],
        dtype=np.float64,
    )


def _visible(observer: object, target_xy: np.ndarray) -> bool:
    relative = _body_relative(observer, target_xy)
    return bool(
        np.linalg.norm(relative) <= 20.0
        and abs(float(atan2(relative[1], relative[0]))) <= pi / 3.0
    )


def assign_decentralized_roles(
    players: Mapping[int, object],
    ball_xy: np.ndarray,
    *,
    side_sign: int,
) -> dict[int, Role]:
    """Infer roles from public field zones, jersey number and each robot's own vision."""

    count = validate_players_per_team(len(players))
    side = 1 if side_sign > 0 else -1
    roles: dict[int, Role] = {1: Role.GOALKEEPER}
    # Every robot can derive the same preferred presser from the visible ball's public
    # y coordinate and the fixed jersey table. No teammate message or global role is needed.
    outfield_count = count - 1
    normalized_ball_y = float(np.clip((float(ball_xy[1]) + 18.0) / 36.0, 0.0, 0.999999))
    preferred_presser = 2 + min(outfield_count - 1, int(normalized_ball_y * outfield_count))
    for player_no in range(2, count + 1):
        agent = players[player_no]
        if not _visible(agent, ball_xy):
            roles[player_no] = Role.COVER
            continue
        own_distance = float(
            np.linalg.norm(np.asarray(agent.xpos[:2], dtype=np.float64) - ball_xy)
        )
        if preferred_presser == player_no:
            roles[player_no] = Role.PRESSER
        else:
            agent_x = float(agent.xpos[0])
            behind_ball = side * (float(ball_xy[0]) - agent_x) >= 0.0
            roles[player_no] = Role.SUPPORTER if behind_ball and own_distance <= 10.0 else Role.COVER
    return roles


def home_position(player_no: int, players_per_team: int, side_sign: int) -> tuple[float, float]:
    """Return a safe formation anchor for any supported 3v3--7v7 team size."""

    count = validate_players_per_team(players_per_team)
    if not 1 <= int(player_no) <= count:
        raise ValueError(f"player_no {player_no} is not active in a {count}-player team")
    if player_no == 1:
        left = (-24.5, 0.0)
    else:
        index = player_no - 2
        outfield = count - 1
        rows = max(1, (outfield + 1) // 2)
        row = index // 2
        lane = 1.0 if index % 2 == 0 else -1.0
        x = -8.0 - 6.0 * (rows - row - 1)
        y = lane * (3.0 + 2.0 * (row % 2))
        if index == outfield - 1 and outfield % 2 == 1:
            y = 0.0
        left = (x, y)
    return (float(left[0] * (1 if side_sign > 0 else -1)), float(left[1]))


def infer_possession_side(
    event: BallEventSnapshot,
    *,
    hold_window_s: float = POSSESSION_HOLD_WINDOW_S,
) -> int:
    """Infer possession strictly from real RCSS contact evidence."""

    return event.contact_backed_possession_side(hold_window_s)


def assign_dynamic_roles(
    players: Mapping[int, object],
    opponents: Mapping[int, object],
    ball_xy: np.ndarray,
    *,
    side_sign: int,
    event: BallEventSnapshot,
    previous_roles: Mapping[int, Role] | None = None,
) -> RoleAssignment:
    """Privileged global role teacher; never feed this assignment to a match Actor."""

    count = validate_players_per_team(len(players))
    side = 1 if side_sign > 0 else -1
    possession = infer_possession_side(event)

    outfield = list(range(2, count + 1))
    presser = min(
        outfield,
        key=lambda number: float(
            np.linalg.norm(np.asarray(players[number].xpos[:2], dtype=np.float64) - ball_xy)
        ),
    )
    roles: dict[int, Role] = {1: Role.GOALKEEPER, presser: Role.PRESSER}
    remaining = [number for number in outfield if number != presser]
    remaining.sort(
        key=lambda number: float(
            np.linalg.norm(np.asarray(players[number].xpos[:2], dtype=np.float64) - ball_xy)
        )
    )
    own_possession = possession == side
    for index, number in enumerate(remaining):
        if own_possession:
            roles[number] = Role.SUPPORTER if index == 0 or index % 2 == 0 else Role.COVER
        else:
            roles[number] = Role.COVER if index == 0 or index % 2 == 0 else Role.SUPPORTER
    previous = {} if previous_roles is None else previous_roles
    changed = sum(previous.get(number) != role for number, role in roles.items())
    return RoleAssignment(roles=roles, possession_side=possession, changed_players=changed)
