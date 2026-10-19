"""v043: parameterized 3v3-to-7v7 recurrent MAPPO team-decision experiment."""

from experiments.v043_team_mappo.spec import (
    ACTION_DIM,
    CENTRAL_STATE_DIM,
    LOCAL_OBSERVATION_DIM,
    MAX_PLAYERS_PER_TEAM,
    Role,
)

__all__ = [
    "ACTION_DIM",
    "CENTRAL_STATE_DIM",
    "LOCAL_OBSERVATION_DIM",
    "MAX_PLAYERS_PER_TEAM",
    "Role",
]
