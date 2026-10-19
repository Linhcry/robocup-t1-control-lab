"""基于 RCSSServerMJ 原生接触状态的足球归属事件跟踪器。

这里不把“足球发生位移”冒充为触球。每个触球、释放和换人触球都直接来自
``SoccerBall.active_contact``、``last_contact`` 与 ``active_contact_body_parts``。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ContactIdentity:
    """可序列化的 RCSS 球员身份。"""

    team_id: int
    player_no: int

    @classmethod
    def from_agent_id(cls, agent_id: object | None) -> ContactIdentity | None:
        if agent_id is None:
            return None
        return cls(team_id=int(getattr(agent_id, "team_id")), player_no=int(getattr(agent_id, "player_no")))


@dataclass(frozen=True, slots=True)
class BallEvent:
    """一次离散的球权事件。"""

    kind: str
    sim_time_s: float
    actor: ContactIdentity | None
    previous_actor: ContactIdentity | None
    body_parts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


class BallEventTracker:
    """把逐物理帧接触转换成不重复的触球、释放与换权事件。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.previous_active: ContactIdentity | None = None
        self.active_body_parts: tuple[str, ...] = ()
        self.last_distinct_toucher: ContactIdentity | None = None
        self.events: list[BallEvent] = []
        self.touch_counts: dict[ContactIdentity, int] = {}
        self.transfer_count = 0
        self.upper_body_contact_count = 0

    @staticmethod
    def _is_upper_body(parts: tuple[str, ...]) -> bool:
        markers = ("hand", "forearm", "upper_arm", "torso", "head")
        return any(any(marker in part.lower() for marker in markers) for part in parts)

    def update(self, ball: object, sim_time_s: float) -> tuple[BallEvent, ...]:
        """读取一帧原生球接触，并返回这一帧新产生的事件。"""

        active = ContactIdentity.from_agent_id(getattr(ball, "active_contact", None))
        parts = tuple(sorted(str(part) for part in (getattr(ball, "active_contact_body_parts", None) or ())))
        frame_events: list[BallEvent] = []

        if active != self.previous_active:
            if self.previous_active is not None:
                frame_events.append(
                    BallEvent("release", float(sim_time_s), self.previous_active, None, ())
                )
            if active is not None:
                previous_toucher = self.last_distinct_toucher
                kind = "touch"
                if previous_toucher is not None and active != previous_toucher:
                    kind = "transfer"
                    self.transfer_count += 1
                event = BallEvent(kind, float(sim_time_s), active, previous_toucher, parts)
                frame_events.append(event)
                self.touch_counts[active] = self.touch_counts.get(active, 0) + 1
                if self._is_upper_body(parts):
                    self.upper_body_contact_count += 1
                self.last_distinct_toucher = active
        elif active is not None and parts != self.active_body_parts:
            # RCSS can keep the same active player while the contacting geometry
            # changes (for example foot -> hand).  Ignoring this transition lets an
            # illegal upper-body touch pass the old edge-only foul gate.
            frame_events.append(
                BallEvent("body_parts_change", float(sim_time_s), active, active, parts)
            )
            if self._is_upper_body(parts) and not self._is_upper_body(self.active_body_parts):
                self.upper_body_contact_count += 1

        self.previous_active = active
        self.active_body_parts = parts if active is not None else ()
        self.events.extend(frame_events)
        return tuple(frame_events)

    def has_lower_leg_contact(self, actor: ContactIdentity) -> bool:
        """Return true only for a current foot/ankle/calf/shin contact by ``actor``."""

        if self.previous_active != actor or self._is_upper_body(self.active_body_parts):
            return False
        lower_leg_markers = ("foot", "ankle", "calf", "shin", "lower_leg")
        return any(
            any(marker in part.lower() for marker in lower_leg_markers)
            for part in self.active_body_parts
        )

    def touched(self, actor: ContactIdentity) -> bool:
        return self.touch_counts.get(actor, 0) > 0

    def transferred(self, passer: ContactIdentity, receiver: ContactIdentity) -> bool:
        return any(
            event.kind == "transfer"
            and event.previous_actor == passer
            and event.actor == receiver
            for event in self.events
        )

    def summary(self) -> dict[str, Any]:
        return {
            "event_count": len(self.events),
            "transfer_count": int(self.transfer_count),
            "upper_body_contact_count": int(self.upper_body_contact_count),
            "last_distinct_toucher": (
                asdict(self.last_distinct_toucher) if self.last_distinct_toucher is not None else None
            ),
            "events": [event.to_dict() for event in self.events],
        }
