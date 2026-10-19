"""局部视觉观测、规则合规的分散式启发策略和动作缩放。

每个球员只能读取自己的 64 维观测。训练器可以汇总七名球员的奖励，
但部署 Actor 不能读取队友的私有状态，这与赛事禁止外部私网通信的要求一致。
"""

from __future__ import annotations

from math import atan2, pi
from typing import Final, Mapping, Sequence

import numpy as np

from locomotion.official_t1_policy import MotionCommand
from locomotion.rcss_simulation import torso_upright


LOCAL_OBS_DIM: Final[int] = 64
ACTION_DIM: Final[int] = 3
VISION_RANGE_M: Final[float] = 20.0
# 前向总视场暂按 120° 近似，因此半角为 60°；赛前应按 TC 最终服务器感知参数复核。
VISION_HALF_ANGLE_RAD: Final[float] = pi / 3.0
MAX_VX: Final[float] = 0.45
MAX_VY: Final[float] = 0.30
MAX_YAW_RATE: Final[float] = 0.20

# 左侧球队的固定 2-2-2 阵型；右侧球队只需把 x 镜像。
LEFT_HOME: Final[dict[int, tuple[float, float]]] = {
    1: (-25.0, 0.0),
    2: (-16.0, 6.5),
    3: (-16.0, -6.5),
    4: (-8.5, 8.5),
    5: (-8.5, -8.5),
    6: (-2.5, 4.0),
    7: (-2.5, -4.0),
}


def wrap_angle(angle: float) -> float:
    """把任意弧度角折回 ``[-pi, pi]``。"""

    return float((angle + pi) % (2.0 * pi) - pi)


def player_yaw(agent: object) -> float:
    """从 MuJoCo 躯干旋转矩阵读取世界坐标系中的偏航角。"""

    rotation = np.asarray(agent.xmat, dtype=np.float64).reshape(3, 3)
    return float(atan2(rotation[1, 0], rotation[0, 0]))


def world_to_body(agent: object, vector_xy: Sequence[float]) -> np.ndarray:
    """把世界坐标二维向量旋转到当前球员机身坐标。"""

    yaw = player_yaw(agent)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    x_world, y_world = float(vector_xy[0]), float(vector_xy[1])
    return np.asarray(
        [cosine * x_world + sine * y_world, -sine * x_world + cosine * y_world],
        dtype=np.float32,
    )


def home_position(player_no: int, side_sign: int) -> tuple[float, float]:
    """``side_sign=+1`` 表示左队向右进攻，``-1`` 表示右队向左进攻。"""

    left_x, y = LEFT_HOME[player_no]
    return (left_x * side_sign, y)


def _visible(relative_body_xy: np.ndarray) -> bool:
    """用有限距离和前向视场近似赛事局部视觉。"""

    distance = float(np.linalg.norm(relative_body_xy))
    angle = abs(float(atan2(relative_body_xy[1], relative_body_xy[0])))
    return distance <= VISION_RANGE_M and angle <= VISION_HALF_ANGLE_RAD


def _encode_visible(relative_body_xy: np.ndarray, half_x: float, half_y: float) -> tuple[float, float, float]:
    """不可见对象写零；可见对象写归一化相对位置和 1.0 掩码。"""

    if not _visible(relative_body_xy):
        return 0.0, 0.0, 0.0
    return (
        float(np.clip(relative_body_xy[0] / half_x, -1.0, 1.0)),
        float(np.clip(relative_body_xy[1] / half_y, -1.0, 1.0)),
        1.0,
    )


def build_local_observation(
    simulation: object,
    agent: object,
    teammates: Mapping[int, object],
    opponents: Mapping[int, object],
    *,
    side_sign: int,
) -> np.ndarray:
    """构造一名球员的 64 维局部观测，具体切片见学习文档。"""

    observation = np.zeros(LOCAL_OBS_DIM, dtype=np.float32)
    half_x = float(simulation.field.field_dim[0]) / 2.0
    half_y = float(simulation.field.field_dim[1]) / 2.0
    player_no = int(agent.agent_id.player_no)
    rotation = np.asarray(agent.xmat, dtype=np.float32).reshape(3, 3)
    body_velocity = rotation.T @ np.asarray(agent._qvel[:3], dtype=np.float32)

    # 0:5 是不依赖视觉的本体感觉。
    observation[0:5] = (
        float(torso_upright(agent)),
        float(np.clip(agent.xpos[2] / 1.2, 0.0, 1.5)),
        float(np.clip(body_velocity[0] / MAX_VX, -2.0, 2.0)),
        float(np.clip(body_velocity[1] / MAX_VY, -2.0, 2.0)),
        float(np.clip(agent._qvel[5] / 2.0, -1.0, 1.0)),
    )

    # 5:10 是球的相对位置、相对速度和可见掩码。
    ball_relative = world_to_body(agent, np.asarray(simulation.ball.xpos[:2]) - np.asarray(agent.xpos[:2]))
    ball_encoded = _encode_visible(ball_relative, half_x, half_y)
    if ball_encoded[2] > 0.5:
        ball_velocity_body = rotation.T @ np.asarray(simulation.ball._qvel[:3], dtype=np.float32)
        observation[5:10] = (
            ball_encoded[0],
            ball_encoded[1],
            float(np.clip(ball_velocity_body[0] / 3.0, -1.0, 1.0)),
            float(np.clip(ball_velocity_body[1] / 3.0, -1.0, 1.0)),
            1.0,
        )

    # 10:14 是公开地图中的对方球门方向和本角色基础站位方向。
    opponent_goal = np.asarray([side_sign * half_x, 0.0])
    home = np.asarray(home_position(player_no, side_sign))
    goal_body = world_to_body(agent, opponent_goal - np.asarray(agent.xpos[:2]))
    home_body = world_to_body(agent, home - np.asarray(agent.xpos[:2]))
    observation[10:14] = (
        float(np.clip(goal_body[0] / half_x, -2.0, 2.0)),
        float(np.clip(goal_body[1] / half_y, -2.0, 2.0)),
        float(np.clip(home_body[0] / half_x, -2.0, 2.0)),
        float(np.clip(home_body[1] / half_y, -2.0, 2.0)),
    )

    # 14:32 固定按球衣号写入其余 6 名队友，每人三个数。
    cursor = 14
    for teammate_no in range(1, 8):
        if teammate_no == player_no:
            continue
        teammate = teammates[teammate_no]
        relative = world_to_body(agent, np.asarray(teammate.xpos[:2]) - np.asarray(agent.xpos[:2]))
        observation[cursor : cursor + 3] = _encode_visible(relative, half_x, half_y)
        cursor += 3

    # 32:53 固定按球衣号写入 7 名对手，每人三个数。
    for opponent_no in range(1, 8):
        opponent = opponents[opponent_no]
        relative = world_to_body(agent, np.asarray(opponent.xpos[:2]) - np.asarray(agent.xpos[:2]))
        observation[cursor : cursor + 3] = _encode_visible(relative, half_x, half_y)
        cursor += 3

    # 53:60 是角色 one-hot；60:64 是阵营、时间、比分和比赛模式。
    observation[53 + player_no - 1] = 1.0
    game_state = simulation.game_state
    own_score = game_state.left_team_score if side_sign > 0 else game_state.right_team_score
    opponent_score = game_state.right_team_score if side_sign > 0 else game_state.left_team_score
    play_modes = list(type(game_state.play_mode))
    play_mode_index = play_modes.index(game_state.play_mode)
    observation[60:64] = (
        float(side_sign),
        float(np.clip(game_state.play_time / 600.0, 0.0, 1.0)),
        float(np.clip((own_score - opponent_score) / 3.0, -1.0, 1.0)),
        float(play_mode_index) / max(1, len(play_modes) - 1),
    )

    if observation.shape != (LOCAL_OBS_DIM,) or not np.isfinite(observation).all():
        raise FloatingPointError("7v7 局部观测必须是有限的 64 维向量")
    return observation


def heuristic_local_action(observation: np.ndarray) -> np.ndarray:
    """只读一名球员观测的规则合规启发式教师，输出三个归一化动作。"""

    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (LOCAL_OBS_DIM,):
        raise ValueError(f"expected ({LOCAL_OBS_DIM},), got {obs.shape}")
    player_no = int(np.argmax(obs[53:60])) + 1
    ball_visible = bool(obs[9] > 0.5)
    ball_body = np.asarray([obs[5] * 27.5, obs[6] * 18.0], dtype=np.float32)
    home_body = np.asarray([obs[12] * 27.5, obs[13] * 18.0], dtype=np.float32)

    # 前锋主动追球；中后场只在球进入较近视野时离开基础站位。
    engage_distance = {1: 3.5, 2: 6.0, 3: 6.0, 4: 8.0, 5: 8.0, 6: 20.0, 7: 20.0}[player_no]
    use_ball = ball_visible and float(np.linalg.norm(ball_body)) <= engage_distance
    target_body = ball_body if use_ball else home_body
    distance = float(np.linalg.norm(target_body))
    target_angle = float(atan2(target_body[1], target_body[0]))

    # 先对准目标再加速，减少 T1 横着走导致的跌倒；数值上限来自 v032 验收。
    forward_scale = np.clip(distance / 2.0, 0.0, 1.0) * np.clip(1.0 - abs(target_angle) / 1.4, 0.15, 1.0)
    lateral_scale = np.clip(target_body[1] / 2.5, -0.65, 0.65)
    yaw_scale = np.clip(target_angle / 0.8, -1.0, 1.0)
    if distance < 0.25 and not use_ball:
        forward_scale = 0.0
        lateral_scale = 0.0
    return np.asarray([forward_scale, lateral_scale, yaw_scale], dtype=np.float32)


def action_to_motion_command(action: np.ndarray, observation: np.ndarray) -> MotionCommand:
    """把神经网络的 ``[-1,1]^3`` 动作映射到 v032 已验收速度范围。"""

    safe_action = np.asarray(action, dtype=np.float32)
    if safe_action.shape != (ACTION_DIM,) or not np.isfinite(safe_action).all():
        raise FloatingPointError("战术动作必须是有限的三维向量")
    safe_action = np.clip(safe_action, -1.0, 1.0)
    ball_angle = float(atan2(observation[6], observation[5])) if observation[9] > 0.5 else 0.0
    return MotionCommand(
        vx=float(MAX_VX * safe_action[0]),
        vy=float(MAX_VY * safe_action[1]),
        yaw_rate=float(MAX_YAW_RATE * safe_action[2]),
        head_yaw=float(np.clip(ball_angle, -0.8, 0.8)),
        head_pitch=0.28,
    )
