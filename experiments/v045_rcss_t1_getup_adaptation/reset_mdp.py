"""Numerically safe fallen-pose resets for exact-RCSS T1 adaptation.

The upstream task drops a fully random articulated robot from 0.8 m.  That is
useful as a broad source-domain curriculum, but the exact RCSS collision model
can very rarely explode after a deeply self-intersecting sample.  v045 instead
trains on the same four pose families used by the formal RCSS evaluator, with
zero initial velocity and immediate policy takeover.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Quaternion order is w, x, y, z.  Pose order is part of the evidence contract.
POSE_NAMES = ("prone", "supine", "left_side", "right_side")
BASE_QUATERNIONS = (
    (0.7071067811865476, 0.0, 0.7071067811865475, 0.0),   # prone
    (0.7071067811865476, 0.0, -0.7071067811865475, 0.0),  # supine
    (0.7071067811865476, -0.7071067811865475, 0.0, 0.0),  # left side
    (0.7071067811865476, 0.7071067811865475, 0.0, 0.0),   # right side
)

# Calibrated by ``v045_reset_height_calibration_run_01.json`` over 5,000 seeded
# samples for each formal RCSS pose.  The domain is +/-0.04 rad orientation and
# +/-0.035 rad joint jitter.  Every fixed height is rounded upward to 1 mm and
# retains more than 10 mm above the largest sampled 5 mm-clearance requirement.
ROOT_HEIGHTS_M = (0.237, 0.135, 0.295, 0.297)

# Non-promotable reverse-curriculum heights calibrated by
# ``v045_tilt_curriculum_height_calibration_run_01.json``.  The formal 90-degree
# stage keeps the larger 5,000-sample heights above instead of the smaller
# 1,000-sample curriculum estimates.
TILT_ROOT_HEIGHTS_M = {
    5: (0.706, 0.698, 0.707, 0.706),
    10: (0.711, 0.695, 0.712, 0.711),
    12: (0.712, 0.693, 0.712, 0.713),
    14: (0.713, 0.690, 0.714, 0.713),
    15: (0.713, 0.689, 0.713, 0.713),
    30: (0.693, 0.644, 0.692, 0.691),
    45: (0.631, 0.564, 0.629, 0.629),
    60: (0.532, 0.443, 0.521, 0.523),
    75: (0.394, 0.301, 0.384, 0.384),
    90: ROOT_HEIGHTS_M,
}

# Backward-compatible read-only names used by older evidence scripts.
_BASE_QUATERNIONS = BASE_QUATERNIONS
_ROOT_HEIGHTS_M = ROOT_HEIGHTS_M


def reset_rcss_four_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    *,
    fall_probability: float = 1.0,
    orientation_jitter_rad: float = 0.04,
    joint_jitter_rad: float = 0.035,
    tilt_deg: int = 90,
    standing_yaw_randomization: bool = False,
    standing_orientation_jitter_rad: float = 0.0,
    standing_joint_jitter_rad: float = 0.0,
    pose_indices: tuple[int, ...] | None = None,
    root_height_drop_jitter_m: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset to a seeded four-pose mixture without a high uncontrolled drop."""

    if not 0.0 <= fall_probability <= 1.0:
        raise ValueError("fall_probability must be in [0, 1]")
    if not 0.0 <= orientation_jitter_rad <= 0.04:
        raise ValueError("orientation_jitter_rad must be in [0, 0.04]")
    if not 0.0 <= joint_jitter_rad <= 0.035:
        raise ValueError("joint_jitter_rad must be in [0, 0.035]")
    if not isinstance(standing_yaw_randomization, bool):
        raise ValueError("standing_yaw_randomization must be boolean")
    if not 0.0 <= standing_orientation_jitter_rad <= 0.04:
        raise ValueError("standing_orientation_jitter_rad must be in [0, 0.04]")
    if not 0.0 <= standing_joint_jitter_rad <= 0.035:
        raise ValueError("standing_joint_jitter_rad must be in [0, 0.035]")
    if tilt_deg not in TILT_ROOT_HEIGHTS_M:
        raise ValueError(
            f"tilt_deg must be one calibrated stage, got {tilt_deg}"
        )
    if not 0.0 <= root_height_drop_jitter_m <= 0.010:
        raise ValueError("root_height_drop_jitter_m must be in [0, 0.010]")
    if pose_indices is None:
        pose_indices = (0, 1, 2, 3)
    if (
        not pose_indices
        or len(set(pose_indices)) != len(pose_indices)
        or any(index not in range(len(POSE_NAMES)) for index in pose_indices)
    ):
        raise ValueError("pose_indices must be unique members of [0, 1, 2, 3]")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    n = int(env_ids.numel())
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    default_joint_vel = asset.data.default_joint_vel
    soft_limits = asset.data.soft_joint_pos_limits
    if default_joint_pos is None or default_joint_vel is None or soft_limits is None:
        raise RuntimeError("RCSS articulation reset tensors are unavailable")

    pose_candidates = torch.as_tensor(
        pose_indices, device=env.device, dtype=torch.int64
    )
    pose_index = pose_candidates[
        torch.randint(0, len(pose_indices), (n,), device=env.device)
    ]
    fall_mask = torch.rand(n, device=env.device) < fall_probability
    pose_axes = torch.as_tensor(
        ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        device=env.device,
        dtype=torch.float32,
    )
    signed_angles = torch.as_tensor(
        (1.0, -1.0, -1.0, 1.0), device=env.device, dtype=torch.float32
    ) * math.radians(tilt_deg)
    base = quat_from_angle_axis(signed_angles, pose_axes)[pose_index]
    heights = torch.as_tensor(
        TILT_ROOT_HEIGHTS_M[tilt_deg], device=env.device, dtype=torch.float32
    )[pose_index]
    if root_height_drop_jitter_m > 0.0:
        heights -= torch.empty(n, device=env.device).uniform_(
            0.0, root_height_drop_jitter_m
        )

    yaw = torch.empty(n, device=env.device).uniform_(-torch.pi, torch.pi)
    jitter_x = torch.empty(n, device=env.device).uniform_(
        -orientation_jitter_rad, orientation_jitter_rad
    )
    jitter_y = torch.empty(n, device=env.device).uniform_(
        -orientation_jitter_rad, orientation_jitter_rad
    )
    x_axis = torch.zeros((n, 3), device=env.device)
    x_axis[:, 0] = 1.0
    y_axis = torch.zeros((n, 3), device=env.device)
    y_axis[:, 1] = 1.0
    z_axis = torch.zeros((n, 3), device=env.device)
    z_axis[:, 2] = 1.0
    orientation = quat_mul(
        quat_from_angle_axis(yaw, z_axis),
        quat_mul(
            quat_from_angle_axis(jitter_y, y_axis),
            quat_mul(quat_from_angle_axis(jitter_x, x_axis), base),
        ),
    )
    orientation = F.normalize(orientation, dim=-1)

    fallen_positions = env.scene.env_origins[env_ids].clone()
    fallen_positions[:, 2] += heights
    default_root_state = asset.data.default_root_state
    if default_root_state is None:
        raise RuntimeError("RCSS default root state is unavailable")
    standing_positions = default_root_state[env_ids, :3].clone()
    standing_positions += env.scene.env_origins[env_ids]
    standing_positions[:, 2] += 0.02
    mask = fall_mask.unsqueeze(-1)
    positions = torch.where(mask, fallen_positions, standing_positions)
    standing_yaw = (
        yaw if standing_yaw_randomization else torch.zeros_like(yaw)
    )
    standing_jitter_x = torch.empty(n, device=env.device).uniform_(
        -standing_orientation_jitter_rad, standing_orientation_jitter_rad
    )
    standing_jitter_y = torch.empty(n, device=env.device).uniform_(
        -standing_orientation_jitter_rad, standing_orientation_jitter_rad
    )
    standing_orientation = quat_mul(
        quat_from_angle_axis(standing_yaw, z_axis),
        quat_mul(
            quat_from_angle_axis(standing_jitter_y, y_axis),
            quat_mul(
                quat_from_angle_axis(standing_jitter_x, x_axis),
                default_root_state[env_ids, 3:7],
            ),
        ),
    )
    standing_orientation = F.normalize(standing_orientation, dim=-1)
    orientation = torch.where(mask, orientation, standing_orientation)
    asset.write_root_link_pose_to_sim(
        torch.cat((positions, orientation), dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(
        torch.zeros((n, 6), device=env.device), env_ids=env_ids
    )

    jitter = torch.empty_like(default_joint_pos[env_ids]).uniform_(
        -joint_jitter_rad, joint_jitter_rad
    )
    fallen_joint_pos = default_joint_pos[env_ids] + jitter
    limits = soft_limits[env_ids]
    fallen_joint_pos = torch.maximum(
        torch.minimum(fallen_joint_pos, limits[..., 1]), limits[..., 0]
    )
    standing_jitter = torch.empty_like(default_joint_pos[env_ids]).uniform_(
        -standing_joint_jitter_rad, standing_joint_jitter_rad
    )
    standing_joint_pos = default_joint_pos[env_ids] + standing_jitter
    standing_joint_pos = torch.maximum(
        torch.minimum(standing_joint_pos, limits[..., 1]), limits[..., 0]
    )
    joint_pos = torch.where(mask, fallen_joint_pos, standing_joint_pos)
    joint_vel = torch.zeros_like(default_joint_vel[env_ids])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # The v045 policy owns the robot from the first control step.
    if "settle_mask" not in env.extras:
        env.extras["settle_mask"] = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    env.extras["settle_mask"][env_ids] = False
    if "v045_reset_pose_index" not in env.extras:
        env.extras["v045_reset_pose_index"] = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.int64
        )
    env.extras["v045_reset_pose_index"][env_ids] = torch.where(
        fall_mask,
        pose_index,
        torch.full_like(pose_index, -1),
    )
    if "v045_reset_is_fallen" not in env.extras:
        env.extras["v045_reset_is_fallen"] = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    env.extras["v045_reset_is_fallen"][env_ids] = fall_mask
