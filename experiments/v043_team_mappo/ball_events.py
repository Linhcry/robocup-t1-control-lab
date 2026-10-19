"""Convert RCSS ball contacts into explicit, testable team events."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rcsssmj.games.teams import TeamSide


POSSESSION_HOLD_WINDOW_S = 1.5
POSSESSION_EVIDENCE_SCHEMA = "rcss_real_touch_plus_finite_hold_v1"


def _signed_team(team_id: int) -> int:
    if int(team_id) == int(TeamSide.LEFT.value):
        return 1
    if int(team_id) == int(TeamSide.RIGHT.value):
        return -1
    return 0


def _contact_key(contact: object | None) -> tuple[int, int] | None:
    if contact is None:
        return None
    return (_signed_team(int(contact.team_id)), int(contact.player_no))


@dataclass(frozen=True)
class BallEventSnapshot:
    """One physics-step event plus persistent last-touch context."""

    active_team: int = 0
    active_player: int = 0
    last_touch_team: int = 0
    last_touch_player: int = 0
    touch_started: bool = False
    teammate_touch_change: bool = False
    turnover: bool = False
    contact_age_s: float = 0.0
    engine_contact_change_s: float | None = None

    def contact_backed_possession_side(
        self,
        hold_window_s: float = POSSESSION_HOLD_WINDOW_S,
    ) -> int:
        """Return ownership only from an active contact or a recent real touch."""

        window = float(hold_window_s)
        if window < 0.0:
            raise ValueError("possession hold window must be non-negative")
        if self.active_team != 0:
            return int(self.active_team)
        if (
            self.last_touch_team != 0
            and 0.0 <= float(self.contact_age_s) <= window
        ):
            return int(self.last_touch_team)
        return 0

    def possession_evidence_source(
        self,
        hold_window_s: float = POSSESSION_HOLD_WINDOW_S,
    ) -> str:
        """Describe the physical evidence supporting the current ownership label."""

        if self.active_team != 0:
            return "active_contact"
        if self.contact_backed_possession_side(hold_window_s) != 0:
            return "recent_touch_hold"
        return "neutral"

    def encoded_for_side(self, side_sign: int) -> tuple[float, ...]:
        """Return privileged team features for the centralized critic/metrics only."""

        side = 1 if side_sign > 0 else -1
        return (
            float(self.active_team == side),
            float(self.last_touch_team == side),
            float(self.last_touch_team == -side),
            float(self.turnover and self.last_touch_team == side),
            float(self.turnover and self.last_touch_team == -side),
            float(self.teammate_touch_change and self.last_touch_team == side),
            min(max(float(self.contact_age_s) / 5.0, 0.0), 1.0),
        )

    def encoded_for_local_actor(
        self,
        side_sign: int,
        player_no: int,
        *,
        ball_visible: bool,
    ) -> tuple[float, ...]:
        """Expose only this robot's own touch state; never broadcast global ownership."""

        side = 1 if side_sign > 0 else -1
        own_active = self.active_team == side and self.active_player == int(player_no)
        own_last = self.last_touch_team == side and self.last_touch_player == int(player_no)
        return (
            float(ball_visible),
            float(own_active),
            float(own_last and ball_visible),
            float(own_active and self.touch_started),
            0.0,
            0.0,
            min(max(float(self.contact_age_s) / 5.0, 0.0), 1.0)
            if own_last and ball_visible
            else 0.0,
        )


class BallEventTracker:
    """Track touch transitions without guessing from ball displacement."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_active: tuple[int, int] | None = None
        self._last_touch: tuple[int, int] | None = None
        self._last_touch_time_s = 0.0
        self.snapshot = BallEventSnapshot()

    def update(self, ball: object, sim_time_s: float) -> BallEventSnapshot:
        """Read ``active_contact``, ``last_contact`` and ``contact_change`` once."""

        now = float(sim_time_s)
        active = _contact_key(ball.active_contact)
        engine_last = _contact_key(ball.last_contact)
        contact_change = ball.contact_change
        previous_touch = self._last_touch
        touch_started = active is not None and active != self._last_active
        teammate_change = False
        turnover = False
        if touch_started:
            if previous_touch is not None:
                teammate_change = active[0] == previous_touch[0] and active[1] != previous_touch[1]
                turnover = active[0] != previous_touch[0]
            self._last_touch = active
        if active is not None:
            # Refresh on every physical-contact frame so the finite hold window starts
            # when contact is released, not when a long contact originally began.
            self._last_touch_time_s = now
        elif (
            self._last_touch is None
            and engine_last is not None
            and contact_change is not None
            and math.isfinite(float(contact_change))
        ):
            # After a tracker reset, preserve the engine's timestamp instead of
            # pretending a possibly stale last contact happened at ``now``.
            self._last_touch = engine_last
            self._last_touch_time_s = min(now, float(contact_change))

        last_touch = self._last_touch
        self.snapshot = BallEventSnapshot(
            active_team=0 if active is None else active[0],
            active_player=0 if active is None else active[1],
            last_touch_team=0 if last_touch is None else last_touch[0],
            last_touch_player=0 if last_touch is None else last_touch[1],
            touch_started=touch_started,
            teammate_touch_change=teammate_change,
            turnover=turnover,
            contact_age_s=0.0 if last_touch is None else max(0.0, now - self._last_touch_time_s),
            engine_contact_change_s=None if contact_change is None else float(contact_change),
        )
        self._last_active = active
        return self.snapshot
