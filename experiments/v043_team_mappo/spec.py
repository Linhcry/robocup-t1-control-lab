"""Stable tensor contracts shared by the v043 environment and MAPPO model."""

from __future__ import annotations

from enum import IntEnum
from typing import Final


MIN_PLAYERS_PER_TEAM: Final[int] = 3
MAX_PLAYERS_PER_TEAM: Final[int] = 7
LOCAL_OBSERVATION_DIM: Final[int] = 92
CENTRAL_STATE_DIM: Final[int] = 96
ACTION_DIM: Final[int] = 3
ENTITY_MASK_DIM: Final[int] = 14
RECURRENT_HIDDEN_DIM: Final[int] = 128


class Role(IntEnum):
    """One dynamic tactical responsibility assigned at each high-level step."""

    GOALKEEPER = 0
    PRESSER = 1
    SUPPORTER = 2
    COVER = 3


ROLE_COUNT: Final[int] = len(Role)


def validate_players_per_team(players_per_team: int) -> int:
    """Reject unsupported sizes before an expensive MuJoCo model is compiled."""

    value = int(players_per_team)
    if not MIN_PLAYERS_PER_TEAM <= value <= MAX_PLAYERS_PER_TEAM:
        raise ValueError(
            f"players_per_team must be in [{MIN_PLAYERS_PER_TEAM}, {MAX_PLAYERS_PER_TEAM}], got {value}"
        )
    return value
