"""创建一台 Booster T1 的 RCSSServerMJ 2026 内存足球仿真。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import rcsssmj.sim.simulation as simulation_module


# rcsssmj 0.2.1 的无音频内存模式会访问空音频对象；运动训练无需音频。
simulation_module.a_forward = lambda *args, **kwargs: None
simulation_module.a_step = lambda *args, **kwargs: None

from rcsssmj.games.soccer.sim.soccer_agent import SoccerAgent  # noqa: E402
from rcsssmj.games.soccer.sim.soccer_sim import SoccerSimulation  # noqa: E402
from rcsssmj.games.soccer.soccer_fields import FIFA7vs7SoccerField  # noqa: E402
from rcsssmj.games.soccer.soccer_rules import SSim2026Rules  # noqa: E402


@dataclass(frozen=True)
class AgentParams:
    """服务器添加一名球员时需要的三项标识。"""

    model_name: str = "T1"
    team_name: str = "QiBlue"
    player_no: int = 1


class SingleT1SoccerSimulation(SoccerSimulation):
    """只用于单机器人训练的稳定重编译版本。

    rcsssmj 0.2.1 在给空球场第一次加入 T1 时会保留旧 ``MjData``，新增执行器的
    ``ctrl`` 因而可能短暂为 NaN，并打印 ``ACTUATOR 6`` 警告。这里在首次装配
    机器人时创建全新的 ``MjData``，先把控制量清零，再计算正向动力学。
    """

    def _recompile(self) -> None:
        """重新编译模型，并保证新加入的全部执行器从有限的零控制量开始。"""

        self._last_recompilation = self._frame_id
        self._mj_model = self._mj_spec.compile()
        self._mj_data = mujoco.MjData(self._mj_model)
        self._mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._a_data = simulation_module.a_recompile(self._mj_spec, self._a_data)
        simulation_module.a_forward(self._a_data, self._mj_data)
        for simulation_object in self.sim_objects:
            simulation_object.bind(self._mj_model, self._mj_data)
        self._update_body_children_map()


def torso_upright(agent: SoccerAgent) -> float:
    """返回躯干局部 z 轴与世界 z 轴的点积：1 表示完全直立。"""

    return float(agent.xmat[8])


def make_single_t1_simulation() -> tuple[SoccerSimulation, SoccerAgent]:
    """创建官方七人制场地，并只加入一台用于低层技能训练的 T1。"""

    simulation = SingleT1SoccerSimulation(FIFA7vs7SoccerField(), SSim2026Rules())
    if not simulation.init():
        raise RuntimeError("RCSSServerMJ 初始化失败")
    players = simulation.add_players([AgentParams()])
    if not players or players[0] is None:
        raise RuntimeError("无法向 RCSSServerMJ 加入 Booster T1")
    simulation.mj_data.ctrl[:] = 0.0
    return simulation, players[0]
