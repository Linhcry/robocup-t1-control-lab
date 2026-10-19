"""3v3-first team training environment with shared rewards and role shaping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rcsssmj.games.soccer.game_phase import GamePhase

from experiments.v043_team_mappo.ball_events import (
    POSSESSION_EVIDENCE_SCHEMA,
    POSSESSION_HOLD_WINDOW_S,
)
from experiments.v043_team_mappo.core import ParameterizedTeamCore, TeamView
from experiments.v043_team_mappo.spec import Role, validate_players_per_team
from locomotion.rcss_simulation import torso_upright


class TeamSelfPlayEnv:
    """Left team learns; right team is a teacher or a frozen league opponent."""

    def __init__(
        self,
        low_level_model: str | Path,
        *,
        players_per_team: int = 3,
        action_repeat: int = 5,
        episode_high_level_steps: int = 300,
        prepare_seconds: float = 0.5,
        seed: int = 43001,
    ) -> None:
        self.players_per_team = validate_players_per_team(players_per_team)
        self.core = ParameterizedTeamCore(
            low_level_model,
            players_per_team=self.players_per_team,
            phase=GamePhase.FIRST_HALF,
        )
        self.action_repeat = int(action_repeat)
        self.episode_high_level_steps = int(episode_high_level_steps)
        self.prepare_seconds = float(prepare_seconds)
        self.rng = np.random.default_rng(seed)
        self.high_level_step = 0
        self.previous_left_score = 0
        self.previous_right_score = 0

    def reset(self) -> tuple[TeamView, TeamView, dict[str, Any]]:
        state = self.core.simulation.game_state
        state.switch_phase(GamePhase.FIRST_HALF, 0.0)
        state.left_team_score = 0
        state.right_team_score = 0
        self.core.simulation.referee.reset()
        self.core.place_teams()
        self.core.prepare(self.prepare_seconds)
        ball_x = float(self.rng.uniform(-2.0, 2.0))
        ball_y = float(self.rng.uniform(-2.0, 2.0))
        self.core.simulation.ball.drop_at(ball_x, ball_y)
        mujoco.mj_forward(self.core.simulation.mj_model, self.core.simulation.mj_data)
        self.core.events.reset()
        self.core.events.update(self.core.simulation.ball, float(self.core.simulation.mj_data.time))
        self.core.simulation.referee.play_on()
        self.core.update_high_level_state()
        self.high_level_step = 0
        self.previous_left_score = 0
        self.previous_right_score = 0
        info = {
            "players_per_team": self.players_per_team,
            "policy_status": "environment_reset_no_neural_claim",
            "ball_start_xy": [ball_x, ball_y],
        }
        return self.core.team_view(1), self.core.team_view(-1), info

    def step(
        self,
        left_actions: np.ndarray,
        right_actions: np.ndarray,
    ) -> tuple[TeamView, TeamView, np.ndarray, bool, dict[str, Any]]:
        ball = self.core.simulation.ball
        old_ball_x = float(ball.xpos[0])
        old_distances = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(self.core.left_players[number].xpos[:2])
                    - np.asarray(ball.xpos[:2])
                )
                for number in range(1, self.players_per_team + 1)
            ],
            dtype=np.float32,
        )
        any_turnover = False
        teammate_touch_change = False
        learner_touch_events = 0
        opponent_touch_events = 0
        for _ in range(self.action_repeat):
            event = self.core.physics_step(left_actions, right_actions)
            any_turnover = any_turnover or event.turnover
            teammate_touch_change = teammate_touch_change or event.teammate_touch_change
            if event.touch_started:
                learner_touch_events += int(event.active_team == 1)
                opponent_touch_events += int(event.active_team == -1)
        self.high_level_step += 1
        self.core.update_high_level_state()

        state = self.core.simulation.game_state
        new_ball_x = float(ball.xpos[0])
        new_distances = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(self.core.left_players[number].xpos[:2])
                    - np.asarray(ball.xpos[:2])
                )
                for number in range(1, self.players_per_team + 1)
            ],
            dtype=np.float32,
        )
        left_goal_delta = int(state.left_team_score) - self.previous_left_score
        right_goal_delta = int(state.right_team_score) - self.previous_right_score
        shared_progress = 1.5 * (new_ball_x - old_ball_x)
        shared_goal = 30.0 * float(left_goal_delta - right_goal_delta)
        possession = self.core.possession_side
        possession_snapshot = self.core.events.snapshot
        possession_source = possession_snapshot.possession_evidence_source()
        shared_possession = 0.02 if possession > 0 else (-0.02 if possession < 0 else 0.0)
        event_reward = 0.0
        if any_turnover:
            event_reward = 0.25 if self.core.events.snapshot.last_touch_team > 0 else -0.25
        if teammate_touch_change and self.core.events.snapshot.last_touch_team > 0:
            event_reward += 0.10

        upright = np.asarray(
            [float(torso_upright(self.core.left_players[number])) for number in range(1, self.players_per_team + 1)],
            dtype=np.float32,
        )
        rewards = np.full(
            self.players_per_team,
            shared_progress + shared_goal + shared_possession + event_reward,
            dtype=np.float32,
        )
        for index, number in enumerate(range(1, self.players_per_team + 1)):
            role = self.core.left_roles[number]
            if role == Role.PRESSER:
                rewards[index] += 0.04 * float(np.clip(old_distances[index] - new_distances[index], -0.5, 0.5))
            rewards[index] += 0.01 * float(np.clip(upright[index], -1.0, 1.0))
            if upright[index] < 0.20:
                rewards[index] -= 1.0
            if self.core.left_role_changed[number]:
                rewards[index] -= 0.002

        self.previous_left_score = int(state.left_team_score)
        self.previous_right_score = int(state.right_team_score)
        done = bool(
            self.high_level_step >= self.episode_high_level_steps
            or left_goal_delta > 0
            or right_goal_delta > 0
        )
        safety = self.core.safety_snapshot()
        left_view = self.core.team_view(1)
        right_view = self.core.team_view(-1)
        info = {
            "ball_progress_m": new_ball_x - old_ball_x,
            "left_score": int(state.left_team_score),
            "right_score": int(state.right_team_score),
            "possession_side": int(possession),
            "possession_evidence_schema": POSSESSION_EVIDENCE_SCHEMA,
            "possession_evidence_source": possession_source,
            "possession_contact_age_s": float(possession_snapshot.contact_age_s),
            "possession_hold_window_s": POSSESSION_HOLD_WINDOW_S,
            "learner_touch_events": learner_touch_events,
            "opponent_touch_events": opponent_touch_events,
            "turnover": bool(any_turnover),
            "teammate_touch_change": bool(teammate_touch_change),
            "left_roles": list(left_view.roles),
            "right_roles": list(right_view.roles),
            "visible_entity_ratio": float(left_view.entity_masks.mean()),
            "fallen_players": int(safety["total_players"] - safety["standing_players"]),
            "all_targets_sent": bool(safety["all_targets_sent"]),
            "safety": safety,
        }
        return left_view, right_view, rewards, done, info
