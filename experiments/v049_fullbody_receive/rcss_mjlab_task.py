"""Exact-RCSS T1 GPU-parallel full-body receive-and-trap task."""

from __future__ import annotations

import copy

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
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

from experiments.v045_rcss_t1_getup_adaptation.rcss_mjlab_task import (
    get_rcss_t1_robot_cfg,
)
from experiments.v049_fullbody_receive import mdp


TASK_ID = "Mjlab-Fullbody-Receive-RCSS-T1-v049"
LEGAL_LOWER_LEG_BODIES = (
    "Shank_Left",
    "left_foot_link",
    "Shank_Right",
    "right_foot_link",
)
# Fail closed: every robot body outside the four explicitly legal lower-leg
# bodies is illegal for ball receiving, including unnamed collision geoms.
ILLEGAL_BALL_CONTACT_BODIES = (
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
    "Hip_Pitch_Left",
    "Hip_Roll_Left",
    "Hip_Yaw_Left",
    "Ankle_Cross_Left",
    "Hip_Pitch_Right",
    "Hip_Roll_Right",
    "Hip_Yaw_Right",
    "Ankle_Cross_Right",
)
DANGEROUS_GROUND_BODIES = (
    "torso",
    "H1",
    "H2",
    "AL3",
    "left_hand_link",
    "AR3",
    "right_hand_link",
)


RCSS_BALL_XML = """
<mujoco model="rcss_soccer_ball">
  <worldbody>
    <body name="ball" pos="0 0 0.11">
      <freejoint name="ball-root"/>
      <geom name="ball" type="sphere" size="0.11" mass="0.41"
            friction="0.4 0.01 0.01" condim="6" priority="1"
            solref="-5000 -20" rgba="1 1 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _configure_soccer_scene(spec: mujoco.MjSpec) -> None:
    spec.option.density = 1.2
    spec.option.viscosity = 0.00002


def get_rcss_ball_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(RCSS_BALL_XML)


def get_rcss_ball_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_rcss_ball_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(2.0, 0.0, mdp.BALL_RADIUS_M)),
    )


def _body_ball_sensor(name: str, bodies: tuple[str, ...]) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="body", pattern=bodies, entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="ball", entity="ball"),
        secondary_policy="error",
        fields=("found", "force", "pos"),
        reduce="maxforce",
        num_slots=1,
        history_length=4,
    )


def _dangerous_ground_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="dangerous_ground_contact",
        primary=ContactMatch(
            mode="body", pattern=DANGEROUS_GROUND_BODIES, entity="robot"
        ),
        secondary=ContactMatch(mode="geom", pattern="terrain", entity=None),
        secondary_policy="error",
        fields=("found", "force", "pos"),
        reduce="maxforce",
        num_slots=1,
        history_length=4,
    )


def _observations(play: bool) -> dict[str, ObservationGroupCfg]:
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.20, n_max=0.20),
            clip=(-8.0, 8.0),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.5, 1.5),
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"biased": True},
            noise=Unoise(n_min=-0.025, n_max=0.025),
            clip=(-4.0, 4.0),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.2, n_max=1.2),
            scale=0.10,
            clip=(-5.0, 5.0),
        ),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action, clip=(-1.0, 1.0)),
        "ball_relative_position": ObservationTermCfg(
            func=mdp.ball_relative_position_b,
            noise=Unoise(n_min=-0.020, n_max=0.020),
            clip=(-5.0, 5.0),
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=2,
        ),
        "ball_relative_velocity": ObservationTermCfg(
            func=mdp.ball_relative_velocity_b,
            noise=Unoise(n_min=-0.12, n_max=0.12),
            clip=(-8.0, 8.0),
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=2,
        ),
    }
    critic_terms = copy.deepcopy(actor_terms)
    for term in critic_terms.values():
        term.noise = None
        term.delay_min_lag = 0
        term.delay_max_lag = 0
        term.delay_update_period = 0
    # v045's privileged critic receives the exact (unbiased) joint position.
    # Keep that shared 23-D slice semantically identical in v049; encoder bias
    # belongs only to the deployable actor stream.
    critic_terms["joint_pos"].params = {"biased": False}
    critic_terms.update(
        {
            "base_linear_velocity_privileged": ObservationTermCfg(
                func=envs_mdp.base_lin_vel, clip=(-6.0, 6.0)
            ),
            "ball_world_state_privileged": ObservationTermCfg(
                func=mdp.privileged_ball_world_state, clip=(-10.0, 10.0)
            ),
            "scenario_privileged": ObservationTermCfg(
                func=mdp.privileged_scenario, clip=(-3.0, 3.0)
            ),
            "receive_state_privileged": ObservationTermCfg(
                func=mdp.privileged_receive_state, clip=(-3.0, 3.0)
            ),
        }
    )
    return {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
            nan_policy="error",
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            nan_policy="error",
        ),
    }


def _metrics() -> dict[str, MetricsTermCfg]:
    funcs = {
        "competitive_receive_success": mdp.metric_success,
        "legal_foot_or_shank_touch": mdp.metric_legal_touch,
        "illegal_robot_ball_touch": mdp.metric_illegal_touch,
        "dangerous_ground_contact": mdp.metric_dangerous_contact,
        "fell": mdp.metric_fell,
        "joint_violation": mdp.metric_joint_violation,
        "torque_violation": mdp.metric_torque_violation,
        "continuous_control_seconds": mdp.metric_hold_seconds,
        "speed_reduction_fraction": mdp.metric_speed_reduction,
        "control_distance_m": mdp.metric_control_distance,
        "peak_illegal_force_n": mdp.metric_peak_illegal_force,
        "peak_dangerous_ground_force_n": mdp.metric_peak_dangerous_force,
        "curriculum_stage": mdp.metric_curriculum_stage,
    }
    return {
        name: MetricsTermCfg(func=func, reduce="last") for name, func in funcs.items()
    }


def rcss_fullbody_receive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    sensors = (
        _body_ball_sensor("legal_lower_leg_ball_contact", LEGAL_LOWER_LEG_BODIES),
        _body_ball_sensor("illegal_robot_ball_contact", ILLEGAL_BALL_CONTACT_BODIES),
        _dangerous_ground_sensor(),
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_rcss_t1_robot_cfg(),
                "ball": get_rcss_ball_cfg(),
            },
            sensors=sensors,
            num_envs=1,
            env_spacing=11.0,
            extent=6.0,
            spec_fn=_configure_soccer_scene,
        ),
        observations=_observations(play),
        actions={
            "joint_pos": mdp.SafeT1JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.55,
                use_default_offset=True,
                preserve_order=True,
            )
        },
        commands={},
        events={
            "reset_receive_scenario": EventTermCfg(
                mode="reset",
                func=mdp.reset_receive_scenario,
                params={
                    "stage_thresholds": mdp.STAGE_THRESHOLDS,
                    "stage_override": 3 if play else None,
                    "evaluation_cell": None,
                },
            ),
            "encoder_bias": EventTermCfg(
                mode="startup",
                func=envs_mdp.dr.encoder_bias,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "bias_range": (-0.012, 0.012),
                },
            ),
        },
        rewards={
            "upright": RewardTermCfg(func=mdp.upright_reward, weight=1.5),
            "intercept_alignment": RewardTermCfg(
                func=mdp.intercept_alignment_reward,
                weight=2.0,
                params={"std_m": 0.32},
            ),
            "approach_control_radius": RewardTermCfg(
                func=mdp.approach_control_radius_reward, weight=0.5
            ),
            "real_legal_contact": RewardTermCfg(
                func=mdp.legal_contact_reward, weight=8.0
            ),
            "speed_reduction": RewardTermCfg(
                func=mdp.speed_reduction_reward, weight=5.0
            ),
            "stable_control": RewardTermCfg(
                func=mdp.stable_control_reward, weight=12.0
            ),
            "receive_success": RewardTermCfg(func=mdp.success_reward, weight=250.0),
            "illegal_ball_contact": RewardTermCfg(
                func=mdp.illegal_contact_cost, weight=-200.0
            ),
            "dangerous_ground_contact": RewardTermCfg(
                func=mdp.danger_cost, weight=-180.0
            ),
            "fall": RewardTermCfg(func=mdp.fall_cost, weight=-150.0),
            "joint_pos_limits": RewardTermCfg(
                func=envs_mdp.joint_pos_limits, weight=-10.0
            ),
            "joint_vel_l2": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-0.002),
            "torque_saturation": RewardTermCfg(
                func=mdp.torque_saturation_cost, weight=-0.20
            ),
            "action_rate_l2": RewardTermCfg(
                func=envs_mdp.action_rate_l2, weight=-0.04
            ),
        },
        terminations={
            "receive_transition": TerminationTermCfg(
                func=mdp.receive_transition, params={"auto_terminate": True}
            ),
            "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        },
        curriculum={
            "incoming_ball_difficulty": CurriculumTermCfg(
                func=mdp.receive_curriculum,
                params={"stage_thresholds": mdp.STAGE_THRESHOLDS},
            )
        },
        metrics=_metrics(),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso",
            distance=4.0,
            elevation=-14.0,
            azimuth=110.0,
        ),
        sim=SimulationCfg(
            nconmax=160,
            njmax=1400,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=6.0,
    )
    if play:
        cfg.observations["actor"].enable_corruption = False
    return cfg


def rcss_fullbody_receive_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.7,
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
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.995,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        seed=49000,
        num_steps_per_env=32,
        max_iterations=4000,
        save_interval=50,
        experiment_name="v049_fullbody_receive",
        run_name="",
        logger="tensorboard",
        wandb_project="rcss_t1_training",
        clip_actions=1.0,
        upload_model=False,
    )


def register_v049_task() -> None:
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=rcss_fullbody_receive_env_cfg(),
        play_env_cfg=rcss_fullbody_receive_env_cfg(play=True),
        rl_cfg=rcss_fullbody_receive_ppo_runner_cfg(),
    )
