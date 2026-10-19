"""Exact-RCSS T1, 23-DoF, GPU-parallel full-body goalkeeper task."""

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
from experiments.v047_fullbody_goalkeeper import mdp


TASK_ID = "Mjlab-Fullbody-Goalkeeper-RCSS-T1-v047"
STAGE_THRESHOLDS = (600 * 32, 1500 * 32, 2600 * 32)

# This is the collision/physics definition from RCSSServerMJ's soccer world.xml.
# The texture is deliberately omitted because it has no effect on dynamics.
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
    """Apply the global air parameters from RCSS soccer/world.xml."""

    spec.option.density = 1.2
    spec.option.viscosity = 0.00002


def get_rcss_ball_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(RCSS_BALL_XML)


def get_rcss_ball_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_rcss_ball_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(5.0, 0.0, 0.11)),
    )


def _contact_sensor(
    name: str,
    robot_geom_names: tuple[str, ...],
    *,
    secondary_geom: str = "ball",
    secondary_entity: str | None = "ball",
) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(
            mode="geom", pattern=robot_geom_names, entity="robot"
        ),
        secondary=ContactMatch(
            mode="geom", pattern=secondary_geom, entity=secondary_entity
        ),
        secondary_policy="error",
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        # Four physics substeps make one 50 Hz policy step.  History prevents a
        # short collision from disappearing between policy observations.
        history_length=4,
    )


def _observations(play: bool) -> dict[str, ObservationGroupCfg]:
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.20, n_max=0.20),
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
        "ball_state_body": ObservationTermCfg(
            func=mdp.ball_state_body,
            noise=Unoise(n_min=-0.015, n_max=0.015),
            clip=(-3.0, 3.0),
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=2,
        ),
        "intercept": ObservationTermCfg(
            func=mdp.intercept_features,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-3.0, 3.0),
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=2,
        ),
        "goal_context": ObservationTermCfg(
            func=mdp.goalkeeper_context,
            clip=(-2.0, 2.0),
        ),
    }
    critic_terms = copy.deepcopy(actor_terms)
    for term in critic_terms.values():
        term.noise = None
        term.delay_min_lag = 0
        term.delay_max_lag = 0
        term.delay_update_period = 0
    # v045's asymmetric critic deliberately uses unbiased joint positions even
    # though its deployable actor uses encoder-biased positions.  Keeping this
    # distinction makes the first 75 critic columns semantically transferable.
    critic_terms["joint_pos"].params = {}
    critic_terms.update(
        {
            "base_lin_vel": ObservationTermCfg(
                func=envs_mdp.base_lin_vel, clip=(-6.0, 6.0)
            ),
            "privileged_scenario": ObservationTermCfg(
                func=mdp.privileged_scenario, clip=(-3.0, 3.0)
            ),
            "privileged_safety": ObservationTermCfg(
                func=mdp.privileged_safety_state, clip=(-2.0, 2.0)
            ),
            "privileged_contact_forces": ObservationTermCfg(
                func=mdp.privileged_contact_forces, clip=(0.0, 5.0)
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
        "competitive_success": mdp.metric_success,
        "ball_blocked": mdp.metric_blocked,
        "conceded": mdp.metric_conceded,
        "legal_touch": mdp.metric_legal_touch,
        "foot_or_calf_touch": mdp.metric_foot_touch,
        "body_touch": mdp.metric_body_touch,
        "head_touch": mdp.metric_head_touch,
        "hand_or_forearm_touch": mdp.metric_hand_touch,
        "fell": mdp.metric_fell,
        "joint_violation": mdp.metric_joint_violation,
        "unsafe_impact": mdp.metric_unsafe,
        "nonfinite": mdp.metric_nonfinite,
        "torque_saturation_fraction": mdp.metric_torque_saturation_fraction,
        "post_block_stand_seconds": mdp.metric_post_block_stand_seconds,
        "peak_head_force_n": mdp.metric_peak_head_force,
        "peak_hand_force_n": mdp.metric_peak_hand_force,
        "curriculum_stage": mdp.metric_curriculum_stage,
    }
    return {
        name: MetricsTermCfg(func=func, reduce="last")
        for name, func in funcs.items()
    }


def rcss_fullbody_goalkeeper_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = get_rcss_t1_robot_cfg()
    ball_cfg = get_rcss_ball_cfg()
    sensors = (
        _contact_sensor(
            "ball_foot_calf_contact",
            ("left_foot", "right_foot", "left_calf", "right_calf"),
        ),
        _contact_sensor(
            "ball_legal_body_contact", ("torso", "left_thigh", "right_thigh")
        ),
        _contact_sensor("ball_head_contact", ("head",)),
        _contact_sensor(
            "ball_hand_forearm_contact",
            ("left_hand", "right_hand", "left_forearm", "right_forearm"),
        ),
        _contact_sensor(
            "head_ground_contact",
            ("head",),
            secondary_geom="terrain",
            secondary_entity=None,
        ),
        _contact_sensor(
            "hand_ground_contact",
            ("left_hand", "right_hand", "left_forearm", "right_forearm"),
            secondary_geom="terrain",
            secondary_entity=None,
        ),
    )

    transition_params = {
        "auto_terminate": True,
        "safe_after_contact_seconds": 0.30,
        "post_block_stand_seconds": 3.0,
        "head_impact_limit_n": 350.0,
        "hand_impact_limit_n": 300.0,
        "joint_limit_tolerance_rad": 1.0e-4,
        "torque_saturation_fraction": 0.995,
    }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg, "ball": ball_cfg},
            sensors=sensors,
            num_envs=1,
            env_spacing=12.0,
            extent=7.0,
            spec_fn=_configure_soccer_scene,
        ),
        observations=_observations(play),
        actions={
            "joint_pos": mdp.T1ClampedJointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                # Match v045's 23-DoF relative target semantics at warm-start.
                # PPO remains free to adapt every output for goalkeeping.
                scale=0.60,
                use_default_offset=True,
                preserve_order=False,
            )
        },
        commands={},
        events={
            "reset_goalkeeper_scenario": EventTermCfg(
                mode="reset",
                func=mdp.reset_goalkeeper_scenario,
                params={
                    "stage_thresholds": STAGE_THRESHOLDS,
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
            "intercept_alignment": RewardTermCfg(
                func=mdp.intercept_alignment_reward,
                weight=2.0,
                params={"std": 0.35},
            ),
            "goal_line_position": RewardTermCfg(
                func=mdp.goal_line_position_reward, weight=0.5
            ),
            "upright": RewardTermCfg(func=mdp.upright_reward, weight=1.0),
            "legal_contact": RewardTermCfg(
                func=mdp.legal_contact_reward, weight=100.0
            ),
            "new_block": RewardTermCfg(func=mdp.new_block_reward, weight=300.0),
            "post_block_standing": RewardTermCfg(
                func=mdp.post_block_standing_reward, weight=4.0
            ),
            "competitive_success": RewardTermCfg(
                func=mdp.competitive_success_reward, weight=500.0
            ),
            "concede": RewardTermCfg(func=mdp.concede_cost, weight=-400.0),
            "unsafe_impact": RewardTermCfg(
                func=mdp.unsafe_impact_cost, weight=-300.0
            ),
            "new_fall": RewardTermCfg(func=mdp.new_fall_cost, weight=-150.0),
            "dangerous_force": RewardTermCfg(
                func=mdp.dangerous_force_cost,
                weight=-20.0,
                params={"head_limit_n": 350.0, "hand_limit_n": 300.0},
            ),
            "torque_saturation": RewardTermCfg(
                func=mdp.torque_saturation_cost,
                weight=-0.20,
                params={"fraction": 0.90},
            ),
            "joint_pos_limits": RewardTermCfg(
                func=envs_mdp.joint_pos_limits, weight=-10.0
            ),
            "joint_vel_l2": RewardTermCfg(
                func=envs_mdp.joint_vel_l2, weight=-0.002
            ),
            "joint_torques_l2": RewardTermCfg(
                func=envs_mdp.joint_torques_l2, weight=-1.0e-5
            ),
            "action_rate_l2": RewardTermCfg(
                func=envs_mdp.action_rate_l2, weight=-0.05
            ),
        },
        terminations={
            "goalkeeper_transition": TerminationTermCfg(
                func=mdp.GoalkeeperTransition,
                params=transition_params,
            ),
            "time_out": TerminationTermCfg(
                func=envs_mdp.time_out, time_out=True
            ),
        },
        curriculum={
            "shot_difficulty": CurriculumTermCfg(
                func=mdp.goalkeeper_curriculum,
                params={"stage_thresholds": STAGE_THRESHOLDS},
            )
        },
        metrics=_metrics(),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso",
            distance=6.0,
            elevation=-12.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=120,
            njmax=1800,
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
    if play:
        cfg.observations["actor"].enable_corruption = False
    return cfg


def rcss_fullbody_goalkeeper_ppo_runner_cfg():
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            # v045's actor consumes these 75 proprioceptive values without an
            # empirical normalizer.  Keeping that contract preserves the exact
            # transferred motor prior; the 12 new bounded ball/context columns
            # start at zero weight and are learned by v047.
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.7,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=8,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.995,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        seed=47000,
        num_steps_per_env=32,
        max_iterations=5000,
        save_interval=50,
        experiment_name="v047_fullbody_goalkeeper",
        run_name="",
        logger="tensorboard",
        wandb_project="rcss_t1_training",
        clip_actions=1.0,
        upload_model=False,
    )


def register_v047_task() -> None:
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=rcss_fullbody_goalkeeper_env_cfg(),
        play_env_cfg=rcss_fullbody_goalkeeper_env_cfg(play=True),
        rl_cfg=rcss_fullbody_goalkeeper_ppo_runner_cfg(),
    )
