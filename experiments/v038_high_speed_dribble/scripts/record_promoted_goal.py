"""只加载通过 held-out test 的 v038 正式晋级模型并录制进球视频。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from experiments.v038_high_speed_dribble.curriculum import DEFAULT_CONFIG, load_config  # noqa: E402
from experiments.v038_high_speed_dribble.high_speed_dribble_env import (  # noqa: E402
    HighSpeedResidualDribbleEnv,
)


CHECKPOINT_PATTERN = re.compile(r"^v038_stage_(\d{2})_iter_(\d{4})$")


@dataclass(frozen=True)
class HeldOutRecord:
    """一个 checkpoint 对应的 held-out test 记录。"""

    checkpoint: Path
    metrics: Path
    stage_index: int
    iteration: int
    quality_passed: bool
    payload: dict[str, Any]


def _portable_name(value: str | Path) -> str:
    """同时识别指标中可能出现的 WSL、Windows 或项目相对路径。"""

    return PurePosixPath(str(value).replace("\\", "/")).name


def _resolve_checkpoint(recorded_path: str | Path) -> Path:
    """把指标内记录的模型路径转换为当前运行环境中真实存在的文件。"""

    raw = Path(recorded_path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(PROJECT_ROOT / raw)
    candidates.append(EXPERIMENT_ROOT / "checkpoints" / _portable_name(recorded_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "held-out 指标指向的 checkpoint 不存在："
        f"{recorded_path}；已检查 {', '.join(str(path) for path in candidates)}"
    )


def _read_held_out_records() -> list[HeldOutRecord]:
    """扫描全部 held-out test；失败记录也保留，以便明确拒绝候选模型。"""

    records: list[HeldOutRecord] = []
    metrics_dir = EXPERIMENT_ROOT / "metrics"
    for metrics_path in sorted(metrics_dir.glob("*_held_out_test.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("split") != "held_out_test":
            continue
        model_value = payload.get("model")
        if not isinstance(model_value, str) or model_value == "zero_residual_teacher":
            continue
        checkpoint_name = _portable_name(model_value)
        match = CHECKPOINT_PATTERN.fullmatch(Path(checkpoint_name).stem)
        if match is None:
            raise ValueError(f"held-out 指标中的模型名不符合 v038 checkpoint 约定：{metrics_path}")
        stage_index = int(match.group(1))
        iteration = int(match.group(2))
        recorded_stage = int(payload.get("stage", {}).get("index", -1))
        if recorded_stage != stage_index:
            raise ValueError(
                f"held-out 指标的 stage 与 checkpoint 不一致：{metrics_path} "
                f"({recorded_stage} != {stage_index})"
            )
        records.append(
            HeldOutRecord(
                checkpoint=_resolve_checkpoint(model_value),
                metrics=metrics_path.resolve(),
                stage_index=stage_index,
                iteration=iteration,
                quality_passed=payload.get("quality_passed") is True,
                payload=payload,
            )
        )
    return records


def select_promoted_checkpoint(requested_model: Path | None = None) -> HeldOutRecord:
    """选择最后一个正式晋级模型；显式模型也必须有通过的 held-out 证据。"""

    records = _read_held_out_records()
    if not records:
        raise FileNotFoundError("v038 metrics 中没有 held_out_test 记录")

    if requested_model is not None:
        requested_name = _portable_name(requested_model)
        matching = [record for record in records if record.checkpoint.name == requested_name]
        if not matching:
            raise RuntimeError(f"拒绝加载 {requested_name}：没有对应的 held-out test 记录")
        promoted = [record for record in matching if record.quality_passed]
        if not promoted:
            failed_metrics = ", ".join(record.metrics.name for record in matching)
            raise RuntimeError(
                f"拒绝加载 {requested_name}：held-out test 未通过（{failed_metrics}）"
            )
        record = max(promoted, key=lambda item: (item.stage_index, item.iteration, item.metrics.name))
        requested_path = requested_model.expanduser()
        if requested_path.is_file() and requested_path.resolve() != record.checkpoint:
            raise RuntimeError(
                f"拒绝加载路径不一致的模型：请求 {requested_path.resolve()}，"
                f"晋级记录指向 {record.checkpoint}"
            )
        return record

    promoted = [record for record in records if record.quality_passed]
    if not promoted:
        raise RuntimeError("v038 没有任何 quality_passed=true 的 held-out test 晋级模型")
    return max(promoted, key=lambda item: (item.stage_index, item.iteration, item.metrics.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, help="可显式选择模型，但仍必须通过 held-out test")
    parser.add_argument("--seed", type=int, default=38016)
    parser.add_argument(
        "--out",
        type=Path,
        default=EXPERIMENT_ROOT / "videos" / "v038_promoted_goal.mp4",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=5)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.video_fps <= 0:
        raise ValueError("width、height 和 video-fps 必须为正数")

    promotion = select_promoted_checkpoint(args.model)
    config = load_config(args.config)
    if not 0 <= promotion.stage_index < len(config["stages"]):
        raise IndexError(f"晋级 stage {promotion.stage_index} 不在当前配置中")

    model = PPO.load(str(promotion.checkpoint), device="cpu")
    env = HighSpeedResidualDribbleEnv(config, promotion.stage_index, seed=args.seed)
    observation, _ = env.reset(seed=args.seed)
    renderer = mujoco.Renderer(env.simulation.mj_model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.2
    camera.azimuth = 40.0
    camera.elevation = -20.0
    render_every = max(1, round(1.0 / (args.video_fps * env.decision_dt)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_reward = 0.0
    final_info: dict[str, object] = {}

    def frame() -> np.ndarray:
        midpoint = 0.58 * np.asarray(env.agent.xpos) + 0.42 * np.asarray(env.simulation.ball.xpos)
        camera.lookat[:] = midpoint
        camera.lookat[2] = 0.45
        renderer.update_scene(env.simulation.mj_data, camera)
        return renderer.render()

    try:
        with imageio.get_writer(args.out, fps=args.video_fps, quality=8, macro_block_size=1) as writer:
            writer.append_data(frame())
            decision_step = 0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(action)
                total_reward += float(reward)
                decision_step += 1
                if decision_step % render_every == 0 or terminated or truncated:
                    writer.append_data(frame())
                if terminated or truncated:
                    break
    finally:
        renderer.close()
        env.close()

    metrics = args.out.with_suffix(".json")
    payload = {
        "model": str(promotion.checkpoint),
        "promotion_metrics": str(promotion.metrics),
        "promotion_split": "held_out_test",
        "promotion_quality_passed": True,
        "stage_index": promotion.stage_index,
        "checkpoint_iteration": promotion.iteration,
        "seed": args.seed,
        "episode_reward": total_reward,
        "video_fps": args.video_fps,
        "frame_size": [args.width, args.height],
        "video": str(args.out.resolve()),
        **final_info,
    }
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(metrics.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
