"""Task-specific MDP terms for v046."""

from .actions import SafeT1JointPositionActionCfg
from .commands import DirectionalPassCommand, DirectionalPassCommandCfg
from .observations import (
    ball_relative_position_b,
    ball_relative_velocity_b,
    contact_state_privileged,
    pass_progress_privileged,
    privileged_ball_world_state,
    root_height_privileged,
    target_direction_b,
    target_distance,
)
from .rewards import (
    approach_ball_reward,
    ball_direction_reward,
    ball_forward_speed_reward,
    ball_progress_reward,
    fall_penalty,
    foot_contact_reward,
    pass_success_reward,
    posture_reward,
    target_corridor_reward,
    torque_usage_l2,
    upper_body_ball_penalty,
)
from .terminations import (
    ball_out_of_bounds,
    fallen,
    hard_joint_limit_violation,
    hard_torque_limit_violation,
    pass_success,
    upper_body_ball_contact,
)

__all__ = [
    "SafeT1JointPositionActionCfg",
    "DirectionalPassCommand",
    "DirectionalPassCommandCfg",
    "ball_relative_position_b",
    "ball_relative_velocity_b",
    "contact_state_privileged",
    "pass_progress_privileged",
    "privileged_ball_world_state",
    "root_height_privileged",
    "target_direction_b",
    "target_distance",
    "approach_ball_reward",
    "ball_direction_reward",
    "ball_forward_speed_reward",
    "ball_progress_reward",
    "fall_penalty",
    "foot_contact_reward",
    "pass_success_reward",
    "posture_reward",
    "target_corridor_reward",
    "torque_usage_l2",
    "upper_body_ball_penalty",
    "ball_out_of_bounds",
    "fallen",
    "hard_joint_limit_violation",
    "hard_torque_limit_violation",
    "pass_success",
    "upper_body_ball_contact",
]
