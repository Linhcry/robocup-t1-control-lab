"""Parameterized RCSS/MuJoCo team core for both 3v3 and 7v7."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rcsssmj.games.soccer.game_phase import GamePhase
from rcsssmj.games.soccer.soccer_fields import FIFA7vs7SoccerField
from rcsssmj.games.soccer.soccer_rules import SSim2026Rules

from experiments.v043_team_mappo.ball_events import BallEventSnapshot, BallEventTracker
from experiments.v043_team_mappo.observations import (
    build_central_state,
    build_local_observation,
    entity_mask,
)
from experiments.v043_team_mappo.roles import (
    RoleAssignment,
    assign_decentralized_roles,
    assign_dynamic_roles,
    home_position,
)
from experiments.v043_team_mappo.spec import ACTION_DIM, Role, validate_players_per_team
from experiments.v043_team_mappo.tactics import action_to_motion_command
from locomotion.rcss_simulation import AgentParams, SingleT1SoccerSimulation, torso_upright
from locomotion.rcss_t1_controller import PREPARE_QPOS, RcssT1Controller
from locomotion.t1_spec import clamp_t1_targets


@dataclass(frozen=True)
class TeamView:
    observations: np.ndarray
    central_state: np.ndarray
    entity_masks: np.ndarray
    roles: tuple[int, ...]


class ParameterizedTeamCore:
    """Create 2N robots while keeping all tensors padded to the seven-player contract."""

    def __init__(
        self,
        low_level_model: str | Path,
        *,
        players_per_team: int = 3,
        left_team_name: str = "Learner",
        right_team_name: str = "Opponent",
        phase: GamePhase = GamePhase.FIRST_HALF,
    ) -> None:
        self.players_per_team = validate_players_per_team(players_per_team)
        self.low_level_model = Path(low_level_model).expanduser().resolve()
        rules = SSim2026Rules()
        self.simulation = SingleT1SoccerSimulation(
            FIFA7vs7SoccerField(),
            rules,
            initial_game_phase=phase,
            initial_play_time=rules.get_start_time_for(phase),
        )
        if not self.simulation.init():
            raise RuntimeError("RCSSServerMJ parameterized team simulation initialization failed")
        self.left_players = self._add_team(left_team_name)
        self.right_players = self._add_team(right_team_name)
        self.controllers = {
            (side, number): RcssT1Controller(self.low_level_model)
            for side in (1, -1)
            for number in range(1, self.players_per_team + 1)
        }
        self.events = BallEventTracker()
        self.left_roles: dict[int, Role] = {}
        self.right_roles: dict[int, Role] = {}
        self.left_teacher_roles: dict[int, Role] = {}
        self.right_teacher_roles: dict[int, Role] = {}
        self.left_role_changed: dict[int, bool] = {}
        self.right_role_changed: dict[int, bool] = {}
        self.possession_side = 0
        self.policy_time = 0.0

    def _add_team(self, team_name: str) -> dict[int, Any]:
        added = self.simulation.add_players(
            [
                AgentParams(model_name="T1", team_name=team_name, player_no=number)
                for number in range(1, self.players_per_team + 1)
            ]
        )
        if len(added) != self.players_per_team or any(player is None for player in added):
            raise RuntimeError(f"{team_name} did not join with {self.players_per_team} complete T1 robots")
        return {int(player.agent_id.player_no): player for player in added if player is not None}

    def place_teams(self) -> None:
        safe_prepare = clamp_t1_targets(PREPARE_QPOS).astype(np.float64)
        for side, players in ((1, self.left_players), (-1, self.right_players)):
            yaw = 0.0 if side > 0 else np.pi
            for number, agent in players.items():
                x, y = home_position(number, self.players_per_team, side)
                agent.drop_at(x, y, yaw)
                agent.init_joints(pos=safe_prepare)
                controller = self.controllers[(side, number)]
                controller.policy.reset()
                controller.filtered_targets[:] = PREPARE_QPOS
                controller.target_was_sent.fill(False)
        self.simulation.ball.drop_at(0.0, 0.0)
        self.simulation.mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.simulation.mj_model, self.simulation.mj_data)
        self.policy_time = 0.0
        self.events.reset()
        self._update_roles()

    def prepare(self, seconds: float) -> None:
        steps = max(0, round(float(seconds) / float(self.simulation.timestep)))
        for _ in range(steps):
            for side, players in ((1, self.left_players), (-1, self.right_players)):
                for number, agent in players.items():
                    self.controllers[(side, number)].prepare_step(self.simulation, agent)
            self.simulation.step([])
        self.events.reset()
        self.events.update(self.simulation.ball, float(self.simulation.mj_data.time))
        self._update_roles()

    def _update_roles(self) -> None:
        ball_xy = np.asarray(self.simulation.ball.xpos[:2], dtype=np.float64)
        event = self.events.snapshot
        left_teacher: RoleAssignment = assign_dynamic_roles(
            self.left_players,
            self.right_players,
            ball_xy,
            side_sign=1,
            event=event,
            previous_roles=self.left_teacher_roles,
        )
        right_teacher: RoleAssignment = assign_dynamic_roles(
            self.right_players,
            self.left_players,
            ball_xy,
            side_sign=-1,
            event=event,
            previous_roles=self.right_teacher_roles,
        )
        left_roles = assign_decentralized_roles(
            self.left_players,
            ball_xy,
            side_sign=1,
        )
        right_roles = assign_decentralized_roles(
            self.right_players,
            ball_xy,
            side_sign=-1,
        )
        self.left_role_changed = {
            number: self.left_roles.get(number) != role for number, role in left_roles.items()
        }
        self.right_role_changed = {
            number: self.right_roles.get(number) != role for number, role in right_roles.items()
        }
        self.left_roles = left_roles
        self.right_roles = right_roles
        self.left_teacher_roles = left_teacher.roles
        self.right_teacher_roles = right_teacher.roles
        self.possession_side = left_teacher.possession_side

    def team_view(self, side_sign: int) -> TeamView:
        side = 1 if side_sign > 0 else -1
        players = self.left_players if side > 0 else self.right_players
        opponents = self.right_players if side > 0 else self.left_players
        roles = self.left_roles if side > 0 else self.right_roles
        changes = self.left_role_changed if side > 0 else self.right_role_changed
        observations = np.stack(
            [
                build_local_observation(
                    self.simulation,
                    players[number],
                    players,
                    opponents,
                    roles,
                    self.events.snapshot,
                    side_sign=side,
                    role_changed=changes[number],
                )
                for number in range(1, self.players_per_team + 1)
            ]
        )
        masks = np.stack([entity_mask(row) for row in observations])
        central = build_central_state(
            self.simulation,
            players,
            opponents,
            roles,
            self.events.snapshot,
            side_sign=side,
            possession_side=self.possession_side,
        )
        return TeamView(
            observations=observations,
            central_state=central,
            entity_masks=masks,
            roles=tuple(int(roles[number]) for number in range(1, self.players_per_team + 1)),
        )

    def physics_step(self, left_actions: np.ndarray, right_actions: np.ndarray) -> BallEventSnapshot:
        for side, players, actions in (
            (1, self.left_players, left_actions),
            (-1, self.right_players, right_actions),
        ):
            safe_actions = np.asarray(actions, dtype=np.float32)
            expected = (self.players_per_team, ACTION_DIM)
            if safe_actions.shape != expected or not np.isfinite(safe_actions).all():
                raise FloatingPointError(f"team actions must be finite {expected}, got {safe_actions.shape}")
            observations = self.team_view(side).observations
            for index, number in enumerate(range(1, self.players_per_team + 1)):
                command = action_to_motion_command(safe_actions[index], observations[index])
                self.controllers[(side, number)].command_step(
                    self.simulation,
                    players[number],
                    command,
                    self.policy_time,
                )
        self.simulation.step([])
        self.policy_time += float(self.simulation.timestep)
        event = self.events.update(self.simulation.ball, float(self.simulation.mj_data.time))
        return event

    def update_high_level_state(self) -> None:
        self._update_roles()

    def safety_snapshot(self) -> dict[str, float | bool | int]:
        players = list(self.left_players.values()) + list(self.right_players.values())
        heights = np.asarray([float(agent.xpos[2]) for agent in players])
        uprights = np.asarray([float(torso_upright(agent)) for agent in players])
        standing = (heights >= 0.34) & (uprights >= 0.20)
        return {
            "finite_qpos": bool(np.isfinite(self.simulation.mj_data.qpos).all()),
            "finite_qvel": bool(np.isfinite(self.simulation.mj_data.qvel).all()),
            "finite_ctrl": bool(np.isfinite(self.simulation.mj_data.ctrl).all()),
            "all_targets_sent": bool(all(controller.all_23_targets_sent for controller in self.controllers.values())),
            "targets_expected": int(2 * self.players_per_team * 23),
            "standing_players": int(standing.sum()),
            "total_players": int(2 * self.players_per_team),
            "minimum_height_m": float(heights.min()),
            "minimum_upright": float(uprights.min()),
        }
