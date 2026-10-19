from __future__ import annotations

from dataclasses import dataclass

from experiments.v040_pass_receive.ball_event_tracker import BallEventTracker, ContactIdentity


@dataclass(frozen=True)
class FakeAgentID:
    team_id: int
    player_no: int


class FakeBall:
    active_contact = None
    last_contact = None
    contact_change = None
    active_contact_body_parts = None


def main() -> None:
    ball = FakeBall()
    tracker = BallEventTracker()
    passer = FakeAgentID(0, 1)
    receiver = FakeAgentID(0, 2)
    ball.active_contact = passer
    ball.active_contact_body_parts = {"left_foot"}
    assert [event.kind for event in tracker.update(ball, 0.1)] == ["touch"]
    assert tracker.update(ball, 0.2) == ()
    ball.active_contact = None
    ball.active_contact_body_parts = None
    assert [event.kind for event in tracker.update(ball, 0.3)] == ["release"]
    ball.active_contact = receiver
    ball.active_contact_body_parts = {"right_foot"}
    assert [event.kind for event in tracker.update(ball, 0.4)] == ["transfer"]
    assert tracker.transferred(ContactIdentity(0, 1), ContactIdentity(0, 2))
    assert tracker.transfer_count == 1
    assert tracker.upper_body_contact_count == 0
    ball.active_contact = None
    tracker.update(ball, 0.5)
    ball.active_contact = passer
    ball.active_contact_body_parts = {"left_hand"}
    tracker.update(ball, 0.6)
    assert tracker.upper_body_contact_count == 1
    ball.active_contact_body_parts = {"left_foot"}
    assert [event.kind for event in tracker.update(ball, 0.7)] == ["body_parts_change"]
    assert tracker.has_lower_leg_contact(ContactIdentity(0, 1))
    ball.active_contact_body_parts = {"left_hand"}
    assert [event.kind for event in tracker.update(ball, 0.8)] == ["body_parts_change"]
    assert tracker.upper_body_contact_count == 2
    assert not tracker.has_lower_leg_contact(ContactIdentity(0, 1))
    print("BallEventTracker edge and body-part tests passed")


if __name__ == "__main__":
    main()
