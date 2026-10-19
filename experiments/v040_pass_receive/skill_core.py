"""v040-v042 共用的少量 T1 足球技能仿真工具。"""

from __future__ import annotations

from math import atan2
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from rcsssmj.games.soccer.game_phase import GamePhase
from rcsssmj.games.soccer.soccer_fields import FIFA7vs7SoccerField
from rcsssmj.games.soccer.soccer_rules import SSim2026Rules

from locomotion.official_t1_policy import MotionCommand
from locomotion.rcss_simulation import AgentParams, SingleT1SoccerSimulation, torso_upright
from locomotion.rcss_t1_controller import PREPARE_QPOS, RcssT1Controller
from locomotion.t1_spec import clamp_t1_targets


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_LOW_LEVEL_MODEL = PROJECT_ROOT.parents[1] / "third_party" / "booster_gym" / "deploy" / "models" / "T1.pt"

MAX_VX = 0.45
MAX_VY = 0.24
MAX_YAW_RATE = 0.22


def world_to_body(agent: object, vector_xy: np.ndarray) -> np.ndarray:
    rotation = np.asarray(agent.xmat, dtype=np.float64).reshape(3, 3)
    vector = np.asarray([float(vector_xy[0]), float(vector_xy[1]), 0.0], dtype=np.float64)
    return (rotation.T @ vector)[:2]


def go_to_world(agent: object, target_xy: np.ndarray, *, speed: float = 0.38) -> np.ndarray:
    """把世界坐标路点转换为安全的三维归一化速度动作。"""

    relative = world_to_body(
        agent,
        np.asarray(target_xy, dtype=np.float64) - np.asarray(agent.xpos[:2], dtype=np.float64),
    )
    distance = float(np.linalg.norm(relative))
    angle = float(atan2(relative[1], relative[0]))
    request = min(float(speed), MAX_VX) * float(np.clip(distance / 0.75, 0.18, 1.0))
    forward = request * float(np.clip(np.cos(angle), 0.0, 1.0))
    lateral = float(np.clip(request * np.sin(angle), -MAX_VY, MAX_VY))
    if distance < 0.12:
        forward = 0.0
        lateral = 0.0
    return np.asarray(
        [forward / MAX_VX, lateral / MAX_VY, np.clip(angle / 0.65, -1.0, 1.0)],
        dtype=np.float32,
    )


class MultiT1SkillCore:
    """创建少量球员并保证所有动作仍经过正式 23DoF 控制链。"""

    def __init__(
        self,
        player_specs: Iterable[AgentParams],
        *,
        low_level_model: str | Path = DEFAULT_LOW_LEVEL_MODEL,
    ) -> None:
        self.simulation = SingleT1SoccerSimulation(
            FIFA7vs7SoccerField(),
            SSim2026Rules(),
            initial_game_phase=GamePhase.FIRST_HALF,
        )
        if not self.simulation.init():
            raise RuntimeError("RCSSServerMJ 技能仿真初始化失败")
        specs = list(player_specs)
        added = self.simulation.add_players(specs)
        if len(added) != len(specs) or any(agent is None for agent in added):
            raise RuntimeError("未能加入技能课程要求的全部 T1")
        self.agents = [agent for agent in added if agent is not None]
        self.controllers = {
            (int(agent.agent_id.team_id), int(agent.agent_id.player_no)): RcssT1Controller(low_level_model)
            for agent in self.agents
        }
        self.sim_dt = float(self.simulation.timestep)
        self.policy_time = 0.0

    def controller_for(self, agent: object) -> RcssT1Controller:
        return self.controllers[(int(agent.agent_id.team_id), int(agent.agent_id.player_no))]

    def reset_players(self, placements: list[tuple[float, float, float]], prepare_seconds: float) -> None:
        if len(placements) != len(self.agents):
            raise ValueError("placements 数量必须与球员数量一致")
        state = self.simulation.game_state
        state.switch_phase(GamePhase.FIRST_HALF, 0.0)
        state.left_team_score = 0
        state.right_team_score = 0
        self.simulation.referee.reset()
        safe_prepare = clamp_t1_targets(PREPARE_QPOS).astype(np.float64)
        for agent, (x, y, yaw) in zip(self.agents, placements, strict=True):
            agent.drop_at(float(x), float(y), float(yaw))
            agent.init_joints(pos=safe_prepare)
            controller = self.controller_for(agent)
            controller.policy.reset()
            controller.filtered_targets[:] = PREPARE_QPOS
            controller.target_was_sent.fill(False)
        self.simulation.mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.simulation.mj_model, self.simulation.mj_data)
        for _ in range(round(float(prepare_seconds) / self.sim_dt)):
            for agent in self.agents:
                self.controller_for(agent).prepare_step(self.simulation, agent)
            self.simulation.step([])
        self.policy_time = 0.0

    def physics_step(self, actions: list[np.ndarray]) -> None:
        if len(actions) != len(self.agents):
            raise ValueError("每名 T1 必须恰好收到一个动作")
        for agent, action in zip(self.agents, actions, strict=True):
            values = np.asarray(action, dtype=np.float32)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise FloatingPointError("高层动作必须是有限的三维数组")
            values = np.clip(values, -1.0, 1.0)
            ball_relative = world_to_body(
                agent,
                np.asarray(self.simulation.ball.xpos[:2]) - np.asarray(agent.xpos[:2]),
            )
            command = MotionCommand(
                vx=float(MAX_VX * values[0]),
                vy=float(MAX_VY * values[1]),
                yaw_rate=float(MAX_YAW_RATE * values[2]),
                head_yaw=float(np.clip(atan2(ball_relative[1], ball_relative[0]), -0.8, 0.8)),
                head_pitch=0.28,
            )
            self.controller_for(agent).command_step(
                self.simulation, agent, command, self.policy_time
            )
        self.simulation.step([])
        self.policy_time += self.sim_dt

    def safety(self) -> dict[str, object]:
        heights = np.asarray([float(agent.xpos[2]) for agent in self.agents])
        uprights = np.asarray([float(torso_upright(agent)) for agent in self.agents])
        return {
            "finite_qpos": bool(np.isfinite(self.simulation.mj_data.qpos).all()),
            "finite_qvel": bool(np.isfinite(self.simulation.mj_data.qvel).all()),
            "finite_ctrl": bool(np.isfinite(self.simulation.mj_data.ctrl).all()),
            "all_23_targets_sent": bool(
                all(controller.all_23_targets_sent for controller in self.controllers.values())
            ),
            "minimum_height_m": float(heights.min()),
            "minimum_upright": float(uprights.min()),
            "fallen": bool(np.any((heights < 0.34) | (uprights < 0.20))),
        }

