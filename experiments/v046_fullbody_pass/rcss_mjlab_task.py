"""GPU-parallel exact-RCSS T1 directional passing task.

This is a new task, not a reinterpretation of v040 evidence.  The robot entity
comes from v045's audited RCSS XML construction; v046 adds one physical ball,
strict contact sensors, deployable actor observations and a privileged critic.
"""

from __future__ import annotations

import mujoco

from experiments.v045_rcss_t1_getup_adaptation.rcss_mjlab_task import (
    get_rcss_t1_robot_cfg,
)
from experiments.v046_fullbody_pass import mdp as pass_mdp
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


TASK_ID = "Mjlab-Directional-Pass-Flat-RCSS-T1-v046"
BALL_RADIUS_M = 0.11
BALL_MASS_KG = 0.43
SUPPORTED_STAGE = "directional_pass"
RECEIVE_STAGE_INTERFACE = "receive_stage_reserved_not_implemented"


def get_soccer_ball_spec() -> mujoco.MjSpec:
    """Create an independently simulated FIFA-size training ball."""

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball", pos=(0.0, 0.0, BALL_RADIUS_M))
    body.add_freejoint(name="ball_joint")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(BALL_RADIUS_M, 0.0, 0.0),
        mass=BALL_MASS_KG,
        friction=(0.55, 0.005, 0.0001),
        condim=6,
        rgba=(0.92, 0.92, 0.92, 1.0),
    )
    return spec


def get_soccer_ball_cfg() -> EntityCfg:
    return EntityCfg(spec_fn=get_soccer_ball_spec)


def _actor_terms() -> dict[str, ObservationTermCfg]:
    return {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"biased": True},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action),
        "ball_relative_position": ObservationTermCfg(
            func=pass_mdp.ball_relative_position_b,
            noise=Unoise(n_min=-0.025, n_max=0.025),
        ),
        "ball_relative_velocity": ObservationTermCfg(
            func=pass_mdp.ball_relative_velocity_b,
            noise=Unoise(n_min=-0.15, n_max=0.15),
        ),
        "target_direction": ObservationTermCfg(
            func=pass_mdp.target_direction_b,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "target_distance": ObservationTermCfg(
            func=pass_mdp.target_distance,
            params={"scale_m": 3.0},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
    }


def _critic_terms() -> dict[str, ObservationTermCfg]:
    # Recreate, rather than alias, actor terms so actor noise never mutates the
    # exact critic stream.
    terms = {
        "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action),
        "ball_relative_position": ObservationTermCfg(
            func=pass_mdp.ball_relative_position_b
        ),
        "ball_relative_velocity": ObservationTermCfg(
            func=pass_mdp.ball_relative_velocity_b
        ),
        "target_direction": ObservationTermCfg(func=pass_mdp.target_direction_b),
        "target_distance": ObservationTermCfg(
            func=pass_mdp.target_distance, params={"scale_m": 3.0}
        ),
        "base_linear_velocity_privileged": ObservationTermCfg(
            func=envs_mdp.base_lin_vel
        ),
        "ball_world_state_privileged": ObservationTermCfg(
            func=pass_mdp.privileged_ball_world_state
        ),
        "contact_state_privileged": ObservationTermCfg(
            func=pass_mdp.contact_state_privileged
        ),
        "pass_progress_privileged": ObservationTermCfg(
            func=pass_mdp.pass_progress_privileged
        ),
        "root_height_privileged": ObservationTermCfg(
            func=pass_mdp.root_height_privileged
        ),
    }
    return terms


def rcss_t1_directional_pass_env_cfg(
    play: bool = False, *, stage: str = SUPPORTED_STAGE
) -> ManagerBasedRlEnvCfg:
    """Build stage one; receive is deliberately an unimplemented interface."""

    if stage != SUPPORTED_STAGE:
        if stage == "receive":
            raise NotImplementedError(
                "v046 reserves the receive-stage observation/evidence contract, "
                "but no receive policy has been trained or accepted"
            )
        raise ValueError(f"unsupported v046 stage: {stage!r}")

    foot_ball_sensor = ContactSensorCfg(
        name="foot_ball_contact",
        primary=ContactMatch(
            mode="geom", pattern=("left_foot", "right_foot"), entity="robot"
        ),
        secondary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
        fields=("found", "force", "pos"),
        reduce="maxforce",
        num_slots=1,
        track_air_time=True,
        history_length=4,
    )
    upper_ball_sensor = ContactSensorCfg(
        name="upper_body_ball_contact",
        primary=ContactMatch(
            # Body matching also covers the unnamed shoulder/upper-arm
            # collision geoms in the RCSS XML. A named-geom-only filter would
            # silently miss those illegal contacts.
            mode="body",
            pattern=(
                "torso",
                "H1",
                "H2",
                "AL1",
                "AL2",
                "AL3",
                "left_hand_link",
                "AR1",
                "AR2",
                "AR3",
                "right_hand_link",
                "Waist",
            ),
            entity="robot",
        ),
        secondary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
        fields=("found", "force", "pos"),
        reduce="maxforce",
        num_slots=1,
        history_length=4,
    )

    observations = {
        "actor": ObservationGroupCfg(
            terms=_actor_terms(), concatenate_terms=True, enable_corruption=True
        ),
        "critic": ObservationGroupCfg(
            terms=_critic_terms(), concatenate_terms=True, enable_corruption=False
        ),
    }
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": pass_mdp.SafeT1JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",), scale=0.6
        )
    }
    commands: dict[str, CommandTermCfg] = {
        "directional_pass": pass_mdp.DirectionalPassCommandCfg(
            resampling_time_range=(8.0, 8.0), debug_vis=play
        )
    }
    events = {
        "reset_robot_root": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"yaw": (-0.15, 0.15)},
                "velocity_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "yaw": (-0.05, 0.05),
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.02, 0.02),
                "velocity_range": (-0.05, 0.05),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "encoder_bias": EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.01, 0.01),
            },
        ),
    }
    rewards = {
        "posture": RewardTermCfg(func=pass_mdp.posture_reward, weight=1.5),
        "approach_ball": RewardTermCfg(
            func=pass_mdp.approach_ball_reward, weight=1.5
        ),
        "true_foot_contact": RewardTermCfg(
            func=pass_mdp.foot_contact_reward, weight=2.0
        ),
        "ball_forward_speed": RewardTermCfg(
            func=pass_mdp.ball_forward_speed_reward, weight=0.5
        ),
        "ball_direction": RewardTermCfg(
            func=pass_mdp.ball_direction_reward, weight=0.25
        ),
        "ball_progress": RewardTermCfg(
            func=pass_mdp.ball_progress_reward, weight=0.25
        ),
        "target_corridor": RewardTermCfg(
            func=pass_mdp.target_corridor_reward, weight=0.1
        ),
        "pass_success": RewardTermCfg(
            func=pass_mdp.pass_success_reward, weight=2.0
        ),
        "upper_body_ball": RewardTermCfg(
            func=pass_mdp.upper_body_ball_penalty, weight=-20.0
        ),
        "fall": RewardTermCfg(func=pass_mdp.fall_penalty, weight=-15.0),
        "joint_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-5.0),
        "torque_usage": RewardTermCfg(func=pass_mdp.torque_usage_l2, weight=-0.01),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "upper_body_ball": TerminationTermCfg(
            func=pass_mdp.upper_body_ball_contact
        ),
        "fallen": TerminationTermCfg(func=pass_mdp.fallen),
        "hard_joint_limit": TerminationTermCfg(
            func=pass_mdp.hard_joint_limit_violation
        ),
        "hard_torque_limit": TerminationTermCfg(
            func=pass_mdp.hard_torque_limit_violation
        ),
        "ball_out_of_bounds": TerminationTermCfg(func=pass_mdp.ball_out_of_bounds),
        "pass_success": TerminationTermCfg(func=pass_mdp.pass_success),
    }
    # Four-stage teacher/student inspiration is intentionally reduced here to
    # the first two *teacher* ideas only: stable approach, then directional
    # kicking.  Student distillation and constrained noisy-perception RL are
    # future versions, not hidden claims in this task.
    curriculum = {
        "approach_weight": CurriculumTermCfg(
            func=envs_mdp.reward_curriculum,
            params={
                "reward_name": "approach_ball",
                "stages": [
                    {"step": 0, "weight": 1.5},
                    {"step": 400 * 24, "weight": 0.8},
                    {"step": 1200 * 24, "weight": 0.3},
                ],
            },
        ),
        "direction_weight": CurriculumTermCfg(
            func=envs_mdp.reward_curriculum,
            params={
                "reward_name": "ball_direction",
                "stages": [
                    {"step": 0, "weight": 0.25},
                    {"step": 400 * 24, "weight": 1.0},
                    {"step": 1200 * 24, "weight": 2.0},
                ],
            },
        ),
        "success_weight": CurriculumTermCfg(
            func=envs_mdp.reward_curriculum,
            params={
                "reward_name": "pass_success",
                "stages": [
                    {"step": 0, "weight": 2.0},
                    {"step": 400 * 24, "weight": 5.0},
                    {"step": 1200 * 24, "weight": 10.0},
                ],
            },
        ),
    }

    if play:
        observations["actor"].enable_corruption = False
        curriculum = {}

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_rcss_t1_robot_cfg(), "ball": get_soccer_ball_cfg()},
            sensors=(foot_ball_sensor, upper_ball_sensor),
            num_envs=1,
            env_spacing=10.5,
            extent=5.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso",
            distance=3.0,
            elevation=-12.0,
            azimuth=115.0,
        ),
        sim=SimulationCfg(
            nconmax=160,
            njmax=700,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=8.0,
    )


def rcss_t1_directional_pass_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.8,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="v046_fullbody_directional_pass",
        wandb_project="rcss_t1_training",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=3000,
    )


def register_v046_task() -> None:
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=rcss_t1_directional_pass_env_cfg(),
        play_env_cfg=rcss_t1_directional_pass_env_cfg(play=True),
        rl_cfg=rcss_t1_directional_pass_ppo_runner_cfg(),
    )
