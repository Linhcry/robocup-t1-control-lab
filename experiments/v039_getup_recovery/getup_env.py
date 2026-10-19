"""MuJoCo 23-DoF recovery environment for Booster T1.

The checked-in Booster Gym locomotion model exposes only twelve leg actuators.
Get-up needs the arms and waist, so this environment loads the full official
``T1_serial.xml`` model, inserts the missing floor geom at load time, and verifies
the complete 23-joint ordering before the first reset.
"""

from __future__ import annotations

import json
from math import cos, radians, sin
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "v039_getup_curriculum.json"
DEFAULT_BOOSTER_GYM_ROOT = PROJECT_ROOT.parents[1] / "third_party" / "booster_gym"

from locomotion.rcss_t1_controller import PREPARE_QPOS  # noqa: E402
from locomotion.t1_spec import (  # noqa: E402
    T1_JOINT_NAMES,
    T1_LOWER_LIMITS,
    T1_UPPER_LIMITS,
    clamp_t1_targets,
)


STATE_JOINT_LIMIT_TOLERANCE_RAD = 2.0e-3
SENT_TARGET_LIMIT_TOLERANCE_RAD = 1.0e-6


def joint_limit_overshoot(qpos: np.ndarray) -> np.ndarray:
    """Return each physical joint state's distance outside the competition limits."""

    values = np.asarray(qpos, dtype=np.float64)
    if values.shape != (23,):
        raise ValueError(f"T1 joint state must be 23-D, got {values.shape}")
    return np.maximum(T1_LOWER_LIMITS - values, 0.0) + np.maximum(
        values - T1_UPPER_LIMITS, 0.0
    )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return json.loads(source.read_text(encoding="utf-8"))


def canonical_joint_name(name: str) -> str:
    return "Head_yaw" if name == "AAHead_yaw" else name


def _quat_wxyz_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return MuJoCo's free-joint quaternion order: ``w, x, y, z``."""

    cr, sr = cos(roll / 2.0), sin(roll / 2.0)
    cp, sp = cos(pitch / 2.0), sin(pitch / 2.0)
    cy, sy = cos(yaw / 2.0), sin(yaw / 2.0)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _fullbody_model_path() -> Path:
    configured = os.environ.get("BOOSTER_GYM_ROOT")
    root = Path(configured).expanduser() if configured else DEFAULT_BOOSTER_GYM_ROOT
    source = root / "resources" / "T1" / "T1_serial.xml"
    if not source.exists():
        raise FileNotFoundError(
            f"Full-body T1 model is missing: {source}. Set BOOSTER_GYM_ROOT if needed."
        )
    return source


def build_fullbody_t1_model(*, physics_dt_s: float = 0.002) -> mujoco.MjModel:
    """Load the full 23-actuator model without modifying the vendored source file."""

    source = _fullbody_model_path()
    xml = source.read_text(encoding="utf-8")
    mesh_dir = (source.parent / "meshes").as_posix()
    xml = xml.replace('meshdir="meshes/"', f'meshdir="{mesh_dir}"', 1)
    if 'geom name="ground"' not in xml:
        floor = (
            '  <worldbody>\n'
            '    <geom name="ground" type="plane" pos="0 0 0" size="0 0 1" '
            'friction="1.0 0.005 0.0001" condim="3" rgba="0.22 0.30 0.22 1"/>'
        )
        xml = xml.replace("  <worldbody>", floor, 1)
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep = float(physics_dt_s)
    return model


class T1GetUpEnv(gym.Env[np.ndarray, np.ndarray]):
    """Goal-conditioned get-up task with pose and physics randomization."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        stage: int = 0,
        config_path: str | Path | None = None,
        fixed_pose: str | None = None,
    ) -> None:
        super().__init__()
        self.config = load_config(config_path)
        self.stage = int(stage)
        stages = {int(item["stage"]): item for item in self.config["stages"]}
        if self.stage not in stages:
            raise ValueError(f"Unknown v039 stage {self.stage}; choices={sorted(stages)}")
        self.stage_cfg = stages[self.stage]
        self.fixed_pose = fixed_pose
        if fixed_pose is not None and fixed_pose not in self.stage_cfg["poses"]:
            raise ValueError(f"Pose {fixed_pose!r} is not part of stage {self.stage}")

        control = self.config["control"]
        env_cfg = self.config["environment"]
        self.decimation = int(control["decimation"])
        self.filter_retention = float(control["action_filter_retention"])
        self.target_scale = np.asarray(control["target_scale_rad"], dtype=np.float64)
        self.kp = np.asarray(control["stiffness"], dtype=np.float64)
        self.kd = np.asarray(control["damping"], dtype=np.float64)
        self.torque_limit = np.asarray(control["torque_limit_nm"], dtype=np.float64)
        for name, array in {
            "target_scale": self.target_scale,
            "stiffness": self.kp,
            "damping": self.kd,
            "torque_limit": self.torque_limit,
        }.items():
            if array.shape != (23,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must contain 23 finite values, got {array.shape}")

        self.model = build_fullbody_t1_model(physics_dt_s=float(control["physics_dt_s"]))
        self.data = mujoco.MjData(self.model)
        self._audit_model_contract()

        self.trunk_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Trunk"
        )
        self.ground_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )
        self.head_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "head"
        )
        self.contact_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (
                "left_foot_link",
                "right_foot_link",
                "left_hand_link",
                "right_hand_link",
            )
        }

        self.episode_steps = max(
            1,
            int(
                round(
                    float(env_cfg["episode_length_s"])
                    / (self.model.opt.timestep * self.decimation)
                )
            ),
        )
        self.hold_steps_required = max(
            1,
            int(
                round(
                    float(env_cfg["stand_hold_s"])
                    / (self.model.opt.timestep * self.decimation)
                )
            ),
        )
        self.settle_physics_steps = max(
            0, int(round(float(env_cfg["settle_time_s"]) / self.model.opt.timestep))
        )

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(23,), dtype=np.float32)
        # projected gravity 3, body velocities 6, height/upright 2, joint state
        # 46, previous action 23, stage one-hot 4, time 1, four contacts 4.
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(89,), dtype=np.float32
        )

        self.filtered_target = clamp_t1_targets(PREPARE_QPOS).astype(np.float64)
        self.previous_action = np.zeros(23, dtype=np.float64)
        self.previous_height = 0.0
        self.previous_upright = 0.0
        self.step_count = 0
        self.stand_hold_steps = 0
        self.pose_name = ""
        self.max_torque_fraction = 0.0
        self.max_contact_force_n = 0.0
        self.target_was_sent = np.zeros(23, dtype=bool)
        self.requested_target_was_clamped = False
        self.sent_target_limit_violation = False
        self.max_sent_target_overshoot_rad = 0.0
        self.initial_joint_limit_violation = False
        self.initial_max_joint_overshoot_rad = 0.0
        self.joint_limit_violation = False
        self.max_joint_overshoot_rad = 0.0
        self.unsafe = False

    def _audit_model_contract(self) -> None:
        if self.model.nu != 23:
            raise AssertionError(f"Full-body get-up model must have 23 actuators, got {self.model.nu}")
        actuator_names = tuple(
            canonical_joint_name(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
                or ""
            )
            for index in range(self.model.nu)
        )
        if actuator_names != T1_JOINT_NAMES:
            raise AssertionError(
                f"T1 actuator order mismatch\nexpected={T1_JOINT_NAMES}\nactual={actuator_names}"
            )
        joint_names = []
        qpos_adrs = []
        dof_adrs = []
        for expected in T1_JOINT_NAMES:
            runtime = "AAHead_yaw" if expected == "Head_yaw" else expected
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, runtime)
            if joint_id < 0:
                raise AssertionError(f"Missing T1 joint {runtime}")
            joint_names.append(canonical_joint_name(runtime))
            qpos_adrs.append(int(self.model.jnt_qposadr[joint_id]))
            dof_adrs.append(int(self.model.jnt_dofadr[joint_id]))
        if tuple(joint_names) != T1_JOINT_NAMES:
            raise AssertionError("T1 full-body joint order is not canonical")
        self.qpos_adrs = np.asarray(qpos_adrs, dtype=np.int32)
        self.dof_adrs = np.asarray(dof_adrs, dtype=np.int32)

    def _root_pose_for(self, pose: str) -> tuple[float, float, float, float]:
        tilt_min, tilt_max = (float(x) for x in self.stage_cfg["tilt_deg"])
        yaw = float(self.np_random.uniform(-np.pi, np.pi))
        magnitude = radians(float(self.np_random.uniform(tilt_min, tilt_max)))
        sign = -1.0 if self.np_random.random() < 0.5 else 1.0
        if pose == "near_fall":
            roll = sign * magnitude if self.np_random.random() < 0.5 else 0.0
            pitch = sign * magnitude if roll == 0.0 else float(self.np_random.uniform(-0.08, 0.08))
            return roll, pitch, yaw, 0.72
        if pose == "left_side":
            return magnitude, float(self.np_random.uniform(-0.10, 0.10)), yaw, 0.44
        if pose == "right_side":
            return -magnitude, float(self.np_random.uniform(-0.10, 0.10)), yaw, 0.44
        if pose == "prone":
            return 0.0, magnitude, yaw, 0.44
        if pose == "supine":
            return 0.0, -magnitude, yaw, 0.44
        raise ValueError(f"Unknown reset pose {pose!r}")

    def _joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[self.qpos_adrs], self.data.qvel[self.dof_adrs]

    def _root_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        rotation = self.data.xmat[self.trunk_body_id].reshape(3, 3)
        projected_gravity = rotation.T @ np.asarray([0.0, 0.0, -1.0])
        body_linear_velocity = rotation.T @ self.data.qvel[:3]
        body_angular_velocity = rotation.T @ self.data.qvel[3:6]
        height = float(self.data.xpos[self.trunk_body_id, 2])
        upright = float(rotation[2, 2])
        return projected_gravity, body_linear_velocity, body_angular_velocity, height, upright

    def _contacts(self) -> tuple[dict[str, bool], float, float]:
        flags = {name: False for name in self.contact_body_ids}
        max_force = 0.0
        head_force = 0.0
        wrench = np.zeros(6, dtype=np.float64)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.ground_geom_id not in (contact.geom1, contact.geom2):
                continue
            other_geom = contact.geom2 if contact.geom1 == self.ground_geom_id else contact.geom1
            other_body = int(self.model.geom_bodyid[other_geom])
            for name, body_id in self.contact_body_ids.items():
                if other_body == body_id:
                    flags[name] = True
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            force = float(np.linalg.norm(wrench[:3]))
            max_force = max(max_force, force)
            if other_geom == self.head_geom_id:
                head_force = max(head_force, force)
        return flags, max_force, head_force

    def _observation(self) -> np.ndarray:
        gravity, lin_vel, ang_vel, height, upright = self._root_features()
        qpos, qvel = self._joint_state()
        midpoint = (T1_LOWER_LIMITS + T1_UPPER_LIMITS) * 0.5
        half_range = np.maximum((T1_UPPER_LIMITS - T1_LOWER_LIMITS) * 0.5, 1e-5)
        normalized_qpos = (qpos - midpoint) / half_range
        stage_one_hot = np.zeros(4, dtype=np.float64)
        stage_one_hot[self.stage] = 1.0
        contacts, _, _ = self._contacts()
        contact_values = np.asarray(
            [
                contacts["left_foot_link"],
                contacts["right_foot_link"],
                contacts["left_hand_link"],
                contacts["right_hand_link"],
            ],
            dtype=np.float64,
        )
        observation = np.concatenate(
            [
                gravity,
                np.clip(lin_vel, -5.0, 5.0),
                np.clip(ang_vel * 0.25, -5.0, 5.0),
                np.asarray([height, upright]),
                normalized_qpos,
                np.clip(qvel * 0.05, -5.0, 5.0),
                self.previous_action,
                stage_one_hot,
                np.asarray([self.step_count / self.episode_steps]),
                contact_values,
            ]
        ).astype(np.float32)
        if observation.shape != self.observation_space.shape:
            raise AssertionError(
                f"v039 observation shape drifted: {observation.shape} != {self.observation_space.shape}"
            )
        return observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        requested_pose = None if options is None else options.get("pose")
        poses = list(self.stage_cfg["poses"])
        self.pose_name = str(
            requested_pose
            or self.fixed_pose
            or poses[int(self.np_random.integers(0, len(poses)))]
        )
        if self.pose_name not in poses:
            raise ValueError(f"Pose {self.pose_name!r} is not allowed in stage {self.stage}")

        mujoco.mj_resetData(self.model, self.data)
        env_cfg = self.config["environment"]
        roll, pitch, yaw, root_height = self._root_pose_for(self.pose_name)
        self.data.qpos[:3] = np.asarray([0.0, 0.0, root_height], dtype=np.float64)
        self.data.qpos[3:7] = _quat_wxyz_from_euler(roll, pitch, yaw)
        joint_noise = float(env_cfg["joint_noise_rad"])
        noisy_joint_pose = PREPARE_QPOS + self.np_random.uniform(
            -joint_noise, joint_noise, size=23
        )
        self.data.qpos[self.qpos_adrs] = clamp_t1_targets(noisy_joint_pose)
        linear_noise = float(env_cfg["root_velocity_noise_mps"])
        angular_noise = float(env_cfg["root_angular_velocity_noise_rps"])
        self.data.qvel[:3] = self.np_random.uniform(-linear_noise, linear_noise, size=3)
        self.data.qvel[3:6] = self.np_random.uniform(-angular_noise, angular_noise, size=3)
        self.data.qvel[self.dof_adrs] = 0.0

        friction_min, friction_max = (float(x) for x in env_cfg["friction_range"])
        friction = float(self.np_random.uniform(friction_min, friction_max))
        self.model.geom_friction[self.ground_geom_id, 0] = friction
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self.pose_name != "near_fall":
            for _ in range(self.settle_physics_steps):
                mujoco.mj_step(self.model, self.data)

        self.filtered_target = clamp_t1_targets(PREPARE_QPOS).astype(np.float64)
        self.previous_action.fill(0.0)
        self.step_count = 0
        self.stand_hold_steps = 0
        self.max_torque_fraction = 0.0
        self.max_contact_force_n = 0.0
        self.target_was_sent.fill(False)
        self.requested_target_was_clamped = False
        self.sent_target_limit_violation = False
        self.max_sent_target_overshoot_rad = 0.0
        initial_qpos, _ = self._joint_state()
        initial_overshoot = joint_limit_overshoot(initial_qpos)
        self.initial_max_joint_overshoot_rad = float(np.max(initial_overshoot))
        self.initial_joint_limit_violation = bool(
            self.initial_max_joint_overshoot_rad > STATE_JOINT_LIMIT_TOLERANCE_RAD
        )
        # The old metric cleared the state violation after the 0.30 s reset settle,
        # hiding an already-invalid initial pose until a later step happened to cross
        # the limit again.  Preserve that reset evidence for the whole episode.
        self.joint_limit_violation = self.initial_joint_limit_violation
        self.max_joint_overshoot_rad = self.initial_max_joint_overshoot_rad
        self.unsafe = not (
            np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        )
        _, _, _, self.previous_height, self.previous_upright = self._root_features()
        observation = self._observation()
        return observation, {
            "stage": self.stage,
            "pose": self.pose_name,
            "floor_friction": friction,
            "model_nu": int(self.model.nu),
            "all_23_targets": False,
            "initial_joint_limit_violation": self.initial_joint_limit_violation,
            "initial_max_joint_overshoot_rad": self.initial_max_joint_overshoot_rad,
        }

    def _apply_action(self, action: np.ndarray) -> None:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (23,):
            raise ValueError(f"v039 action must be 23-D, got {values.shape}")
        if not np.isfinite(values).all():
            raise FloatingPointError("v039 refuses NaN/Inf actions")
        values = np.clip(values, -1.0, 1.0)
        requested_target = PREPARE_QPOS + values * self.target_scale
        safe_target = clamp_t1_targets(requested_target).astype(np.float64)
        self.requested_target_was_clamped |= bool(
            np.any(np.abs(requested_target - safe_target) > SENT_TARGET_LIMIT_TOLERANCE_RAD)
        )
        self.filtered_target = (
            self.filter_retention * self.filtered_target
            + (1.0 - self.filter_retention) * safe_target
        )
        self.filtered_target = clamp_t1_targets(self.filtered_target).astype(np.float64)
        sent_target_overshoot = joint_limit_overshoot(self.filtered_target)
        current_sent_target_overshoot = float(np.max(sent_target_overshoot))
        self.max_sent_target_overshoot_rad = max(
            self.max_sent_target_overshoot_rad, current_sent_target_overshoot
        )
        self.sent_target_limit_violation |= bool(
            current_sent_target_overshoot > SENT_TARGET_LIMIT_TOLERANCE_RAD
        )
        qpos, qvel = self._joint_state()
        torques = self.kp * (self.filtered_target - qpos) - self.kd * qvel
        torques = np.clip(torques, -self.torque_limit, self.torque_limit)
        self.data.ctrl[:] = torques
        self.target_was_sent[:] = np.isfinite(self.filtered_target)
        self.max_torque_fraction = max(
            self.max_torque_fraction,
            float(np.max(np.abs(torques) / np.maximum(self.torque_limit, 1e-6))),
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_values = np.asarray(action, dtype=np.float64)
        self._apply_action(action_values)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        gravity, lin_vel, ang_vel, height, upright = self._root_features()
        del gravity
        qpos, qvel = self._joint_state()
        contacts, contact_force, head_force = self._contacts()
        self.max_contact_force_n = max(self.max_contact_force_n, contact_force)
        pose_error = float(np.mean(np.abs(qpos - PREPARE_QPOS)))
        root_speed = float(np.linalg.norm(lin_vel))
        foot_contact_count = int(contacts["left_foot_link"]) + int(
            contacts["right_foot_link"]
        )

        env_cfg = self.config["environment"]
        standing_now = bool(
            height >= float(env_cfg["success_height_m"])
            and upright >= float(env_cfg["success_upright_cos"])
            and pose_error <= float(env_cfg["success_joint_mean_error_rad"])
            and root_speed <= float(env_cfg["success_root_speed_mps"])
            and foot_contact_count == 2
        )
        self.stand_hold_steps = self.stand_hold_steps + 1 if standing_now else 0
        success = self.stand_hold_steps >= self.hold_steps_required

        current_joint_overshoot = float(np.max(joint_limit_overshoot(qpos)))
        self.max_joint_overshoot_rad = max(
            self.max_joint_overshoot_rad, current_joint_overshoot
        )
        actual_limit_violation = (
            current_joint_overshoot > STATE_JOINT_LIMIT_TOLERANCE_RAD
        )
        self.joint_limit_violation |= actual_limit_violation
        finite = bool(
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(self.data.ctrl).all()
        )
        unsafe_force = head_force > float(env_cfg["unsafe_contact_force_n"])
        unsafe_joint_overshoot = current_joint_overshoot > float(
            env_cfg["unsafe_joint_overshoot_rad"]
        )
        self.unsafe |= bool(not finite or unsafe_force or unsafe_joint_overshoot)

        upward_progress = height - self.previous_height
        upright_progress = upright - self.previous_upright
        upright_score = float(np.clip((upright - 0.20) / 0.75, 0.0, 1.0))
        height_score = float(np.clip((height - 0.15) / 0.48, 0.0, 1.0))
        standing_quality = upright_score * height_score
        pose_score = float(np.exp(-pose_error / 0.45))
        torque_penalty = float(
            np.mean(np.square(self.data.ctrl / np.maximum(self.torque_limit, 1e-6)))
        )
        action_rate_penalty = float(np.mean(np.square(action_values - self.previous_action)))
        joint_velocity_penalty = float(np.mean(np.square(np.clip(qvel * 0.05, -3.0, 3.0))))
        reward = (
            3.5 * standing_quality
            + 0.55 * pose_score * standing_quality
            + 0.18 * foot_contact_count * standing_quality
            + 10.0 * upward_progress
            + 5.0 * upright_progress
            - 0.20 * (1.0 - standing_quality)
            - 0.06 * torque_penalty
            - 0.02 * action_rate_penalty
            - 0.02 * joint_velocity_penalty
            - 3.0 * current_joint_overshoot
        )
        if success:
            reward += 50.0
        if self.unsafe:
            reward -= 50.0

        self.previous_action = np.clip(action_values, -1.0, 1.0)
        self.previous_height = height
        self.previous_upright = upright
        terminated = bool(success or self.unsafe)
        truncated = bool(self.step_count >= self.episode_steps and not terminated)
        observation = self._observation()
        info: dict[str, Any] = {
            "stage": self.stage,
            "pose": self.pose_name,
            "is_success": success,
            "unsafe": self.unsafe,
            "height_m": height,
            "upright_cos": upright,
            "joint_mean_error_rad": pose_error,
            "root_speed_mps": root_speed,
            "foot_contacts": foot_contact_count,
            "stand_hold_s": self.stand_hold_steps
            * self.model.opt.timestep
            * self.decimation,
            "elapsed_s": self.step_count * self.model.opt.timestep * self.decimation,
            "max_torque_fraction": self.max_torque_fraction,
            "max_contact_force_n": self.max_contact_force_n,
            "initial_joint_limit_violation": self.initial_joint_limit_violation,
            "initial_max_joint_overshoot_rad": self.initial_max_joint_overshoot_rad,
            "joint_limit_violation": self.joint_limit_violation,
            "max_joint_overshoot_rad": self.max_joint_overshoot_rad,
            "requested_target_was_clamped": self.requested_target_was_clamped,
            "sent_target_limit_violation": self.sent_target_limit_violation,
            "max_sent_target_overshoot_rad": self.max_sent_target_overshoot_rad,
            "all_23_targets": bool(self.target_was_sent.all()),
            "finite": finite,
        }
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        return None
