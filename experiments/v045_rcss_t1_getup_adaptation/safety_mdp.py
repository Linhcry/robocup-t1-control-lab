"""Safety-shaped MJLab terms for exact-RCSS T1 getup adaptation."""

from __future__ import annotations

import torch
from mjlab.entity import Entity

from experiments.v044_mjlab_t1_getup.rcss_getup_controller import (
    RCSS_EFFECTIVE_EFFORT_LIMIT,
)


def _sensor_peak_force(env: object, name: str) -> torch.Tensor:
    sensor = env.scene[name]
    history = sensor.data.force_history
    if history is not None:
        return torch.linalg.vector_norm(history, dim=-1).amax(dim=(1, 2))
    force = sensor.data.force
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.linalg.vector_norm(force, dim=-1).amax(dim=1)


def dangerous_ground_force_cost(
    env: object, force_limit_n: float
) -> torch.Tensor:
    """Penalize only the portion of head/hand/torso impact above the contract."""

    if force_limit_n <= 0.0:
        raise ValueError("force_limit_n must be positive")
    peak = torch.maximum(
        _sensor_peak_force(env, "head_ground_contact"),
        torch.maximum(
            _sensor_peak_force(env, "hand_ground_contact"),
            _sensor_peak_force(env, "torso_ground_contact"),
        ),
    )
    return torch.clamp(torch.relu(peak - force_limit_n) / force_limit_n, max=10.0)


def torque_saturation_cost(env: object, fraction: float) -> torch.Tensor:
    """Penalize reliance on the exact per-joint RCSS mechanical caps."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("torque saturation fraction must be in (0, 1)")
    robot: Entity = env.scene["robot"]
    caps = torch.as_tensor(
        RCSS_EFFECTIVE_EFFORT_LIMIT,
        device=env.device,
        dtype=robot.data.qfrc_actuator.dtype,
    ).view(1, -1)
    utilization = robot.data.qfrc_actuator.abs() / caps
    return (
        torch.relu(utilization - fraction) / (1.0 - fraction)
    ).mean(dim=1)


def raw_action_l2_cost(env: object, clip: float = 10.0) -> torch.Tensor:
    """Penalize a large actor mean/std before the deployment rate limiter.

    A policy can otherwise learn very large, nearly constant outputs because
    every value beyond the rate guard produces the same 0.06 rad target step.
    The exact RCSS controller then spends most of its time torque-saturated.
    Keeping this cost on the raw action gives PPO a graded signal even while
    the safety guard is active.
    """

    if clip <= 0.0:
        raise ValueError("raw action cost clip must be positive")
    action_term = env.action_manager.get_term("joint_pos")
    raw_action = torch.clamp(action_term.raw_action, min=-clip, max=clip)
    return torch.square(raw_action).mean(dim=1)


def processed_delta_l2_cost(env: object) -> torch.Tensor:
    """Penalize reliance on the full per-decision target-rate allowance."""

    action_term = env.action_manager.get_term("joint_pos")
    normalized = action_term.processed_delta / 0.06
    return torch.square(normalized).mean(dim=1)


def normalized_joint_velocity_l2_cost(
    env: object, velocity_scale_rad_s: float = 10.0, clip: float = 5.0
) -> torch.Tensor:
    """Penalize continuous joint motion without allowing spikes to dominate."""

    if velocity_scale_rad_s <= 0.0 or clip <= 0.0:
        raise ValueError("joint velocity cost parameters must be positive")
    robot: Entity = env.scene["robot"]
    normalized = torch.clamp(
        robot.data.joint_vel / velocity_scale_rad_s,
        min=-clip,
        max=clip,
    )
    return torch.square(normalized).mean(dim=1)


def formal_stable_stand_reward(
    env: object,
    desired_height_m: float,
    upright_score_min: float,
    root_speed_mps_max: float,
) -> torch.Tensor:
    """Dense version of the formal standing gate, paid on every stable step.

    A brief pass through an upright pose is not enough: the reward is largest
    only when the torso is high and upright, root translation is slow, and both
    feet touch the pitch.  Repeating this reward every step makes sustained
    balance more valuable than a momentary recovery spike.
    """

    if desired_height_m <= 0.0 or not 0.0 < upright_score_min < 1.0:
        raise ValueError("formal stand height/upright thresholds are invalid")
    if root_speed_mps_max <= 0.0:
        raise ValueError("formal stand speed threshold must be positive")
    robot: Entity = env.scene["robot"]
    torso_height = robot.data.root_link_pos_w[:, 2]
    torso_upright = -robot.data.projected_gravity_b[:, 2]
    root_speed = torch.linalg.vector_norm(
        robot.data.root_link_vel_w[:, :3], dim=-1
    )
    left_found = env.scene["left_foot_ground_contact"].data.found
    right_found = env.scene["right_foot_ground_contact"].data.found
    if left_found is None or right_found is None:
        raise RuntimeError("formal stand foot-contact sensors are unavailable")
    both_feet = (
        torch.any(left_found > 0, dim=-1)
        & torch.any(right_found > 0, dim=-1)
    ).float()

    # Smooth gates preserve a learning signal just below the formal threshold,
    # while the two-foot multiplier keeps one-foot fly-throughs unattractive.
    height_score = torch.sigmoid((torso_height - desired_height_m) / 0.04)
    upright_score = torch.sigmoid(
        (torso_upright - upright_score_min) / 0.04
    )
    speed_score = torch.exp(-torch.square(root_speed / root_speed_mps_max))
    feet_score = 0.1 + 0.9 * both_feet
    return height_score * upright_score * speed_score * feet_score
