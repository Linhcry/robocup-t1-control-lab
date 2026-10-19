"""Competition recovery supervisor compatible with Booster's published RoboCup flow.

This module deliberately does not contain a fake get-up trajectory.  On real Booster
hardware, ``request_get_up`` must call the SDK's ``LocoApiId::kGetUp``.  In MuJoCo,
the callback can instead select a policy trained by this experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Callable


class RecoveryState(IntEnum):
    """States published by the Booster recovery channel."""

    READY = 0
    FALLING = 1
    FALLEN = 2
    GETTING_UP = 3


class RecoveryDecision(str, Enum):
    """One deterministic output for every supervisor update."""

    CONTINUE_PLAY = "continue_play"
    WAIT = "wait"
    REQUEST_GET_UP = "request_get_up"
    SAFE_STOP = "safe_stop"


@dataclass
class OfficialRecoverySupervisor:
    """Debounce get-up requests, enforce retries, and block locomotion while fallen."""

    request_get_up: Callable[[], None]
    retry_max_count: int = 2
    recovery_requested: bool = False
    retry_count: int = 0
    previous_mode_index: int | None = None

    def reset(self) -> None:
        self.recovery_requested = False
        self.retry_count = 0
        self.previous_mode_index = None

    def update(
        self,
        *,
        state: RecoveryState,
        recovery_available: bool,
        robot_mode_index: int,
        under_penalty: bool = False,
    ) -> RecoveryDecision:
        """Return a safe action without ever repeating a request every control tick."""

        if under_penalty or robot_mode_index == 2:
            self.reset()
            return RecoveryDecision.SAFE_STOP

        if state == RecoveryState.READY:
            self.recovery_requested = False
            self.retry_count = 0
            self.previous_mode_index = robot_mode_index
            return RecoveryDecision.CONTINUE_PLAY

        if state == RecoveryState.FALLING:
            self.previous_mode_index = robot_mode_index
            return RecoveryDecision.SAFE_STOP

        # The official demo observes mode 10 after a failed recovery attempt.  Count
        # only an edge into that mode, otherwise one failure would consume all retries.
        if (
            self.recovery_requested
            and robot_mode_index == 10
            and self.previous_mode_index != 10
        ):
            self.retry_count += 1
            self.recovery_requested = False

        self.previous_mode_index = robot_mode_index

        if not recovery_available:
            return RecoveryDecision.SAFE_STOP
        if self.retry_count >= self.retry_max_count:
            return RecoveryDecision.SAFE_STOP
        if state == RecoveryState.GETTING_UP or self.recovery_requested:
            return RecoveryDecision.WAIT
        if state == RecoveryState.FALLEN and robot_mode_index == 1:
            self.request_get_up()
            self.recovery_requested = True
            return RecoveryDecision.REQUEST_GET_UP
        return RecoveryDecision.WAIT
