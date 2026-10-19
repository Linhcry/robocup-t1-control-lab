"""MJLab getup task using the exact RCSSServerMJ T1 geometry and joint physics."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import register_mjlab_task

from mjlab_playground.getup.config.t1.env_cfgs import booster_t1_getup_env_cfg
from mjlab_playground.getup.config.t1.rl_cfg import booster_t1_getup_ppo_runner_cfg

from experiments.v045_rcss_t1_getup_adaptation import safety_mdp
from experiments.v045_rcss_t1_getup_adaptation import reset_mdp
from experiments.v045_rcss_t1_getup_adaptation import training_profiles
from experiments.v045_rcss_t1_getup_adaptation.actuator_contract import (
    V045_ACTUATOR_GROUPS,
)
from experiments.v045_rcss_t1_getup_adaptation.safe_actions import (
    RcssSafeRelativeJointPositionActionCfg,
)


TASK_ID = "Mjlab-Getup-Flat-RCSS-T1"


def _find_rcss_t1_xml() -> Path:
    configured = os.environ.get("RCSS_T1_XML")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        Path("/home/linh/robot_soccer/envs/rl-mujoco").glob(
            "lib/python*/site-packages/rcsssmj/resources/robots/T1/robot.xml"
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "RCSS T1 robot.xml not found; set RCSS_T1_XML to the installed model"
    )


RCSS_T1_XML = _find_rcss_t1_xml()


def get_rcss_t1_spec() -> mujoco.MjSpec:
    """Load the competition robot and remove only its predeclared control ports."""

    spec = mujoco.MjSpec.from_file(str(RCSS_T1_XML))
    # RCSS declares tau/position/velocity actuator triplets.  MJLab owns the
    # training actuator, so keeping those ports would double-control each joint.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.665),
    joint_pos={
        "Left_Shoulder_Roll": -1.4,
        "Left_Elbow_Yaw": -0.4,
        "Right_Shoulder_Roll": 1.4,
        "Right_Elbow_Yaw": 0.4,
        ".*_Hip_Pitch": -0.2,
        ".*_Knee_Pitch": 0.4,
        ".*_Ankle_Pitch": -0.2,
    },
    joint_vel={".*": 0.0},
)


def _position_actuator(
    names: tuple[str, ...], stiffness: float, damping: float, effort: float
) -> BuiltinPositionActuatorCfg:
    return BuiltinPositionActuatorCfg(
        target_names_expr=names,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort,
        # These are the exact passive values in the RCSS competition XML.
        armature=0.01,
        frictionloss=0.1,
    )


RCSS_ARTICULATION = EntityArticulationInfoCfg(
    actuators=tuple(
        _position_actuator(names, stiffness, damping, effort)
        for names, stiffness, damping, effort in V045_ACTUATOR_GROUPS
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_rcss_t1_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        spec_fn=get_rcss_t1_spec,
        articulation=RCSS_ARTICULATION,
        # No collision editor: retain RCSS collision shapes, friction and solvers.
        collisions=(),
    )


def _ground_contact_sensor(
    name: str, geom_names: tuple[str, ...]
) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="terrain", entity=None),
        secondary_policy="error",
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        history_length=4,
    )


def rcss_t1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = booster_t1_getup_env_cfg(play=play)
    cfg.scene.entities = {"robot": get_rcss_t1_robot_cfg()}

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    safety_contact_cfgs = (
        _ground_contact_sensor("head_ground_contact", ("head",)),
        _ground_contact_sensor(
            "hand_ground_contact",
            ("left_hand", "right_hand", "left_forearm", "right_forearm"),
        ),
        _ground_contact_sensor("torso_ground_contact", ("torso",)),
        _ground_contact_sensor("left_foot_ground_contact", ("left_foot",)),
        _ground_contact_sensor("right_foot_ground_contact", ("right_foot",)),
    )
    cfg.scene.sensors = tuple(
        sensor for sensor in (cfg.scene.sensors or ())
        if sensor.name
        not in {
            "self_collision",
            "head_ground_contact",
            "hand_ground_contact",
            "torso_ground_contact",
            "left_foot_ground_contact",
            "right_foot_ground_contact",
        }
    ) + (self_collision_cfg, *safety_contact_cfgs)
    cfg.rewards["self_collisions"].params["sensor_name"] = "self_collision"
    cfg.rewards["unsafe_ground_impact"] = RewardTermCfg(
        func=safety_mdp.dangerous_ground_force_cost,
        weight=-2.0,
        params={"force_limit_n": 900.0},
    )
    cfg.rewards["torque_saturation"] = RewardTermCfg(
        func=safety_mdp.torque_saturation_cost,
        weight=-0.05,
        params={"fraction": 0.98},
    )
    # These terms are disabled in the historical profiles.  The v3 standing
    # curriculum enables them explicitly so old evidence remains interpretable.
    cfg.rewards["raw_action_l2"] = RewardTermCfg(
        func=safety_mdp.raw_action_l2_cost,
        weight=0.0,
        params={"clip": 10.0},
    )
    cfg.rewards["processed_delta_l2"] = RewardTermCfg(
        func=safety_mdp.processed_delta_l2_cost,
        weight=0.0,
        params={},
    )
    cfg.rewards["normalized_joint_velocity_l2"] = RewardTermCfg(
        func=safety_mdp.normalized_joint_velocity_l2_cost,
        weight=0.0,
        params={"velocity_scale_rad_s": 10.0, "clip": 5.0},
    )
    cfg.rewards["formal_stable_stand"] = RewardTermCfg(
        func=safety_mdp.formal_stable_stand_reward,
        weight=5.0,
        params={
            "desired_height_m": 0.60,
            "upright_score_min": 0.90,
            "root_speed_mps_max": 0.45,
        },
    )
    cfg.rewards["posture"].params["orientation_threshold"] = 0.05

    cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=("torso",)
    )
    cfg.rewards["waist_height"].params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=("Waist",)
    )
    cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=("torso",)
    )
    cfg.viewer.body_name = "torso"

    # Use deployable proprioception instead of source-model-specific IMU names.
    for group_name in ("actor", "critic"):
        angular = cfg.observations[group_name].terms["base_ang_vel"]
        angular.func = envs_mdp.base_ang_vel
        angular.params = {}
        # Stop at the exact observation term that first becomes non-finite.
        # Sanitizing a physically invalid state would silently poison PPO.
        cfg.observations[group_name].nan_policy = "error"
        cfg.observations[group_name].nan_check_per_term = True
    linear = cfg.observations["critic"].terms["base_lin_vel"]
    linear.func = envs_mdp.base_lin_vel
    linear.params = {}

    # The exact RCSS contact parameters stay fixed during initial adaptation.
    # Encoder bias remains active; the explicit training profile below selects
    # whether torso-COM randomization is enabled for an engineering A/B run.
    cfg.events.pop("geom_friction_slide", None)
    cfg.events.pop("foot_friction_spin", None)
    cfg.events.pop("foot_friction_roll", None)
    # Use the four-pose RCSS contract and avoid the upstream 0.8 m,
    # full-joint-range random drop that produced a rare physical NaN at scale.
    reset = cfg.events["reset_fallen_or_standing"]
    reset.func = reset_mdp.reset_rcss_four_pose
    reset.params = {
        "fall_probability": 1.0,
        "orientation_jitter_rad": 0.04,
        "joint_jitter_rad": 0.035,
    }
    # The official actor output remains raw.  Both training and deployment apply
    # the same 0.6 rad relative delta once per 50 Hz decision, clamp the final
    # absolute target to the deployment contract, then hold it for four substeps.
    cfg.actions["joint_pos"] = RcssSafeRelativeJointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=0.6,
    )
    cfg.terminations["energy"].params["settle_steps"] = 0
    # The task's safe default is the exact formal RCSS solver/reset profile with
    # target COM.  Engineering A/B runs may override these through train_v045.py.
    training_profiles.apply_training_profiles(cfg)
    return cfg


def rcss_t1_getup_ppo_runner_cfg():
    cfg = booster_t1_getup_ppo_runner_cfg()
    cfg.experiment_name = "v045_rcss_t1_getup"
    cfg.wandb_project = "rcss_t1_training"
    # The official T1 get-up task leaves the Gaussian actor output unbounded.
    # Safety is enforced after the 0.6 rad relative action is converted into an
    # absolute target: RcssSafeRelativeJointPositionAction clamps that target to
    # the exact RCSS mechanical joint limits.
    cfg.clip_actions = None
    return cfg


def register_v045_task() -> None:
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=rcss_t1_getup_env_cfg(),
        play_env_cfg=rcss_t1_getup_env_cfg(play=True),
        rl_cfg=rcss_t1_getup_ppo_runner_cfg(),
    )
