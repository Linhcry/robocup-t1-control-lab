"""在实时 MuJoCo Viewer 中加载并展示 v036、v037 或 v038 训练模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from experiments.v036_single_robot_score.single_robot_score_env import SingleRobotScoreEnv  # noqa: E402
from experiments.v036_single_robot_score.training import (  # noqa: E402
    DEFAULT_CONFIG as V036_DEFAULT_CONFIG,
    latest_checkpoint as latest_v036_checkpoint,
    load_config as load_v036_config,
)
from experiments.v037_midfield_dribble_score.curriculum import (  # noqa: E402
    DEFAULT_CONFIG as V037_DEFAULT_CONFIG,
    load_config as load_v037_config,
)
from experiments.v037_midfield_dribble_score.midfield_dribble_env import (  # noqa: E402
    ResidualMidfieldDribbleScoreEnv,
)
from experiments.v037_midfield_dribble_score.residual_curriculum import (  # noqa: E402
    latest_residual_checkpoint,
)
from experiments.v038_high_speed_dribble.curriculum import (  # noqa: E402
    DEFAULT_CONFIG as V038_DEFAULT_CONFIG,
    EXPERIMENT_ROOT as V038_EXPERIMENT_ROOT,
    load_config as load_v038_config,
)
from experiments.v038_high_speed_dribble.high_speed_dribble_env import (  # noqa: E402
    HighSpeedResidualDribbleEnv,
)


V038_CHECKPOINT_PATTERN = re.compile(r"v038_stage_(\d+)_iter_(\d+)\.zip")


def _v038_checkpoint_stage_and_iteration(checkpoint: Path) -> tuple[int, int]:
    """从 v038 检查点文件名读取阶段和迭代编号。"""

    match = V038_CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(
            "v038 模型文件名必须形如 v038_stage_00_iter_0002.zip，"
            f"当前为: {checkpoint.name}"
        )
    return int(match.group(1)), int(match.group(2))


def latest_strict_v038_checkpoint() -> tuple[Path, int, dict[str, Any], Path]:
    """只从严格通过 held-out 全部门槛的记录中选最新 v038 模型。"""

    candidates: list[tuple[int, int, str, Path, dict[str, Any], Path]] = []
    metrics_folder = V038_EXPERIMENT_ROOT / "metrics"
    checkpoints_folder = V038_EXPERIMENT_ROOT / "checkpoints"
    for metrics_path in metrics_folder.glob("*_held_out_test*.json"):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        gates = payload.get("gate_results")
        if (
            payload.get("split") != "held_out_test"
            or payload.get("quality_passed") is not True
            or not isinstance(gates, dict)
            or not gates
            or any(result is not True for result in gates.values())
        ):
            continue

        raw_model = payload.get("model")
        summary = payload.get("summary")
        stage = payload.get("stage")
        if (
            not isinstance(raw_model, str)
            or not isinstance(summary, dict)
            or not isinstance(stage, dict)
        ):
            continue
        # 记录可能由 Windows 或 WSL 写入；统一两类路径分隔符后只取文件名。
        checkpoint_name = raw_model.replace("\\", "/").rsplit("/", 1)[-1]
        checkpoint = checkpoints_folder / checkpoint_name
        try:
            checkpoint_stage, iteration = _v038_checkpoint_stage_and_iteration(checkpoint)
            summary_stage = int(summary["stage_index"])
            recorded_stage = int(stage["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if checkpoint_stage != summary_stage or checkpoint_stage != recorded_stage:
            continue
        candidates.append(
            (checkpoint_stage, iteration, metrics_path.name, checkpoint, payload, metrics_path)
        )

    if not candidates:
        raise FileNotFoundError(
            "没有找到严格通过 held-out_test 且全部 gate 为 true 的 v038 封存模型"
        )
    _, _, _, checkpoint, payload, metrics_path = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"最新通过封存测试的记录存在，但对应检查点缺失: {checkpoint}"
        )
    return checkpoint, int(payload["summary"]["stage_index"]), payload, metrics_path


def _successful_v038_demo_seed(payload: dict[str, Any]) -> int:
    """优先使用该封存测试中已实际进球且未跌倒的第一个种子。"""

    episodes = payload.get("episodes", [])
    if isinstance(episodes, list):
        for episode in episodes:
            if (
                isinstance(episode, dict)
                and episode.get("scored") is True
                and episode.get("official_goal") is True
                and episode.get("fell") is False
            ):
                return int(episode["seed"])
    seeds = payload.get("evaluation_seeds", [])
    if isinstance(seeds, list) and seeds:
        return int(seeds[0])
    raise ValueError("通过的 v038 held-out 记录中没有可用于展示的种子")


def build_experiment(
    experiment: str,
    *,
    model_path: Path | None,
    config_path: Path | None,
    seed: int | None,
) -> tuple[PPO, Any, np.ndarray, Path, int]:
    """加载指定检查点并创建与它匹配的单机器人环境。"""

    if experiment == "v036":
        checkpoint = model_path or latest_v036_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError("没有找到 v036 检查点，请先确认 checkpoints 目录")
        chosen_seed = 36010 if seed is None else int(seed)
        config = load_v036_config(config_path or V036_DEFAULT_CONFIG)
        environment = SingleRobotScoreEnv(config, seed=chosen_seed)
    elif experiment == "v037":
        checkpoint = model_path or latest_residual_checkpoint(4)
        if checkpoint is None:
            raise FileNotFoundError("没有找到 v037 中场阶段检查点，请先确认 checkpoints 目录")
        chosen_seed = 37410 if seed is None else int(seed)
        config = load_v037_config(config_path or V037_DEFAULT_CONFIG)
        environment = ResidualMidfieldDribbleScoreEnv(config, 4, seed=chosen_seed)
    elif experiment == "v038":
        config = load_v038_config(config_path or V038_DEFAULT_CONFIG)
        if model_path is None:
            checkpoint, stage_index, held_out_payload, _ = latest_strict_v038_checkpoint()
            chosen_seed = (
                _successful_v038_demo_seed(held_out_payload) if seed is None else int(seed)
            )
        else:
            checkpoint = model_path
            stage_index, _ = _v038_checkpoint_stage_and_iteration(Path(checkpoint))
            if not 0 <= stage_index < len(config["stages"]):
                raise IndexError(
                    f"v038 模型 stage {stage_index} 不在当前配置的阶段范围内"
                )
            default_seeds = config["stages"][stage_index].get("evaluation_seeds", [])
            if seed is None and not default_seeds:
                raise ValueError(f"v038 stage {stage_index} 没有默认展示种子")
            chosen_seed = int(default_seeds[0]) if seed is None else int(seed)
        environment = HighSpeedResidualDribbleEnv(config, stage_index, seed=chosen_seed)
    else:
        raise ValueError(f"不支持的实验版本: {experiment}")

    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint}")
    policy = PPO.load(str(checkpoint), device="cpu")
    observation, _ = environment.reset(seed=chosen_seed)
    return policy, environment, observation, Path(checkpoint).resolve(), chosen_seed


def update_follow_camera(viewer: Any, environment: Any, experiment: str) -> None:
    """让自由相机跟随机器人和球之间的区域，同时保留鼠标旋转和缩放。"""

    robot_weight = 0.62 if experiment == "v036" else 0.58
    ball_weight = 1.0 - robot_weight
    midpoint = (
        robot_weight * np.asarray(environment.agent.xpos, dtype=np.float64)
        + ball_weight * np.asarray(environment.simulation.ball.xpos, dtype=np.float64)
    )
    viewer.cam.lookat[:] = midpoint
    viewer.cam.lookat[2] = 0.45


def show_stage_overlay(viewer: Any, environment: Any, experiment: str) -> None:
    """在 MuJoCo 窗口左上角持续标出当前课程阶段。"""

    stage_index = getattr(environment, "stage_index", None)
    if stage_index is None:
        return
    stage = getattr(environment, "stage", {})
    stage_name = str(stage.get("name", "未命名阶段")) if isinstance(stage, dict) else "未命名阶段"
    viewer.set_texts(
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            f"{experiment} live | STAGE {stage_index}",
            stage_name,
        )
    )


def run_live_viewer(
    *,
    policy: PPO,
    environment: Any,
    observation: np.ndarray,
    experiment: str,
    speed: float,
    hold_seconds: float,
    follow_camera: bool,
) -> dict[str, Any]:
    """按真实时间推进环境，直到进球、跌倒、超时或用户关闭窗口。"""

    import mujoco.viewer

    if speed <= 0.0:
        raise ValueError("--speed 必须大于 0")
    viewer = mujoco.viewer.launch_passive(
        environment.simulation.mj_model,
        environment.simulation.mj_data,
    )
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 4.0 if experiment == "v036" else 4.2
    viewer.cam.azimuth = 40.0
    viewer.cam.elevation = -18.0 if experiment == "v036" else -20.0
    show_stage_overlay(viewer, environment, experiment)
    if follow_camera:
        update_follow_camera(viewer, environment, experiment)
    viewer.sync()

    total_reward = 0.0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    closed_by_user = False
    target_wall_step = float(environment.decision_dt) / speed

    try:
        while not (terminated or truncated):
            if not viewer.is_running():
                closed_by_user = True
                break
            step_start = time.perf_counter()
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, final_info = environment.step(action)
            total_reward += float(reward)
            if follow_camera:
                update_follow_camera(viewer, environment, experiment)
            viewer.sync()
            remaining = target_wall_step - (time.perf_counter() - step_start)
            if remaining > 0.0:
                time.sleep(remaining)

        hold_deadline = time.perf_counter() + max(0.0, float(hold_seconds))
        while not closed_by_user and viewer.is_running() and time.perf_counter() < hold_deadline:
            viewer.sync()
            time.sleep(0.02)
    finally:
        viewer.close()

    result = {
        "experiment": experiment,
        "episode_reward": total_reward,
        "closed_by_user": closed_by_user,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        **final_info,
    }
    stage_index = getattr(environment, "stage_index", None)
    if stage_index is not None:
        result["viewer_stage_index"] = int(stage_index)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("v036", "v037", "v038"), default="v037")
    parser.add_argument("--model", type=Path, help="可选：显式指定 PPO zip 检查点")
    parser.add_argument("--config", type=Path, help="可选：显式指定该版本 JSON 配置")
    parser.add_argument("--seed", type=int, help="不填写时使用已验收的成功展示种子")
    parser.add_argument("--speed", type=float, default=1.0, help="仿真播放速度，1.0 表示实时")
    parser.add_argument("--hold-seconds", type=float, default=8.0, help="结束后窗口保留秒数")
    parser.add_argument(
        "--no-follow-camera",
        action="store_true",
        help="关闭自动跟随；可在 Viewer 中完全手动控制相机",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查模型、环境和首次推理，不打开图形窗口",
    )
    args = parser.parse_args()

    policy, environment, observation, checkpoint, seed = build_experiment(
        args.experiment,
        model_path=args.model,
        config_path=args.config,
        seed=args.seed,
    )
    header = {
        "experiment": args.experiment,
        "model": str(checkpoint),
        "seed": seed,
        "speed": args.speed,
        "viewer": not args.check_only,
    }
    if args.experiment == "v038" and args.model is None:
        header["model_selection"] = "latest_strict_held_out_pass"
    stage_index = getattr(environment, "stage_index", None)
    if stage_index is not None:
        header["stage_index"] = int(stage_index)
        stage = getattr(environment, "stage", {})
        if isinstance(stage, dict):
            header["stage_name"] = str(stage.get("name", "未命名阶段"))
    print(json.dumps(header, ensure_ascii=False, indent=2), flush=True)

    try:
        if args.check_only:
            action, _ = policy.predict(observation, deterministic=True)
            if np.asarray(action).shape != (3,) or not np.isfinite(action).all():
                raise FloatingPointError("首次策略输出必须是有限的三维动作")
            print("模型加载、环境重置和首次推理检查通过。", flush=True)
            return
        result = run_live_viewer(
            policy=policy,
            environment=environment,
            observation=observation,
            experiment=args.experiment,
            speed=args.speed,
            hold_seconds=args.hold_seconds,
            follow_camera=not args.no_follow_camera,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
