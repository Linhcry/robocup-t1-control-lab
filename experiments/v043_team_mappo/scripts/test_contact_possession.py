"""Regression tests for contact-only v043 possession and promotion evidence."""

from __future__ import annotations

from types import SimpleNamespace

from rcsssmj.games.teams import TeamSide

from experiments.v043_team_mappo.ball_events import (
    POSSESSION_EVIDENCE_SCHEMA,
    POSSESSION_HOLD_WINDOW_S,
    BallEventSnapshot,
    BallEventTracker,
)
from experiments.v043_team_mappo.promotion import _recompute_episode_evidence
from experiments.v043_team_mappo.roles import infer_possession_side


class FakeBall:
    active_contact = None
    last_contact = None
    contact_change = None


def contact(side: TeamSide, player_no: int) -> SimpleNamespace:
    return SimpleNamespace(team_id=int(side.value), player_no=player_no)


def valid_evaluation_row() -> dict[str, object]:
    return {
        "seed": 1,
        "high_level_steps": 10,
        "left_score": 0,
        "right_score": 0,
        "learner_possession_ratio": 0.6,
        "possession_evidence_schema": POSSESSION_EVIDENCE_SCHEMA,
        "possession_hold_window_s": POSSESSION_HOLD_WINDOW_S,
        "possession_sample_steps": 10,
        "learner_contact_possession_steps": 3,
        "opponent_contact_possession_steps": 2,
        "neutral_possession_steps": 5,
        "active_contact_possession_steps": 2,
        "recent_touch_hold_possession_steps": 3,
        "learner_touch_event_count": 1,
        "opponent_touch_event_count": 1,
        "maximum_fallen_players": 0,
        "all_targets_sent": True,
        "simulated_seconds": 1.0,
        "true_pass_attempts": 0,
        "true_pass_completions": 0,
        "opponent_id": "test-opponent",
    }


def main() -> None:
    # No contact means neutral even if some robot would be geometrically nearest.
    assert infer_possession_side(BallEventSnapshot()) == 0
    assert infer_possession_side(
        BallEventSnapshot(last_touch_team=1, contact_age_s=POSSESSION_HOLD_WINDOW_S)
    ) == 1
    assert infer_possession_side(
        BallEventSnapshot(last_touch_team=1, contact_age_s=POSSESSION_HOLD_WINDOW_S + 0.001)
    ) == 0

    # A long active contact refreshes the timestamp; the bounded hold begins on release.
    ball = FakeBall()
    tracker = BallEventTracker()
    ball.active_contact = contact(TeamSide.LEFT, 2)
    assert tracker.update(ball, 1.0).contact_backed_possession_side() == 1
    assert tracker.update(ball, 4.0).contact_age_s == 0.0
    ball.active_contact = None
    recent = tracker.update(ball, 4.1)
    assert recent.contact_backed_possession_side() == 1
    assert recent.possession_evidence_source() == "recent_touch_hold"
    expired = tracker.update(ball, 4.0 + POSSESSION_HOLD_WINDOW_S + 0.001)
    assert expired.contact_backed_possession_side() == 0
    assert expired.possession_evidence_source() == "neutral"

    # Reset recovery may read RCSS ``last_contact``, but must retain its old
    # timestamp rather than minting a fresh possession window.
    stale_ball = FakeBall()
    stale_ball.last_contact = contact(TeamSide.LEFT, 3)
    stale_ball.contact_change = 1.0
    stale = BallEventTracker().update(stale_ball, 10.0)
    assert stale.contact_age_s == 9.0
    assert stale.contact_backed_possession_side() == 0

    row = valid_evaluation_row()
    evaluation = {
        "possession_evidence_schema": POSSESSION_EVIDENCE_SCHEMA,
        "possession_hold_window_s": POSSESSION_HOLD_WINDOW_S,
        "evaluation_seeds": [1],
        "episodes": [row],
        "summary": {},
    }
    assert _recompute_episode_evidence(evaluation)["records_valid"]

    # A nearest-player-style ratio without physical touch counts must fail closed.
    forged = dict(row)
    forged["learner_possession_ratio"] = 1.0
    forged["learner_contact_possession_steps"] = 10
    forged["opponent_contact_possession_steps"] = 0
    forged["neutral_possession_steps"] = 0
    forged["active_contact_possession_steps"] = 0
    forged["recent_touch_hold_possession_steps"] = 10
    forged["learner_touch_event_count"] = 0
    evaluation["episodes"] = [forged]
    assert not _recompute_episode_evidence(evaluation)["records_valid"]

    # One real touch cannot justify an unbounded number of later hold samples.
    unbounded_hold = valid_evaluation_row()
    unbounded_hold.update(
        {
            "high_level_steps": 100,
            "learner_possession_ratio": 1.0,
            "possession_sample_steps": 100,
            "learner_contact_possession_steps": 100,
            "opponent_contact_possession_steps": 0,
            "neutral_possession_steps": 0,
            "active_contact_possession_steps": 0,
            "recent_touch_hold_possession_steps": 100,
            "learner_touch_event_count": 1,
            "opponent_touch_event_count": 0,
            "simulated_seconds": 10.0,
        }
    )
    evaluation["episodes"] = [unbounded_hold]
    assert not _recompute_episode_evidence(evaluation)["records_valid"]
    print("v043 contact-only possession regression passed")


if __name__ == "__main__":
    main()
