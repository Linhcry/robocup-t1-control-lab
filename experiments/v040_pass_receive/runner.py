"""v040-v042 课程共用的确定性评估与 PPO 短训练工具。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def unique_output(path: str | Path) -> Path:
    """历史证据只追加；目标存在时自动使用 ``_run_02``。"""

    requested = Path(path)
    if not requested.exists():
        return requested
    index = 2
    while True:
        candidate = requested.with_name(f"{requested.stem}_run_{index:02d}{requested.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def next_iteration_path(directory: str | Path, prefix: str, suffix: str) -> tuple[int, Path]:
    """返回严格递增的模型/指标路径，不覆盖任何历史迭代。"""

    folder = Path(directory)
    pattern = re.compile(rf"^{re.escape(prefix)}_iter_(\d{{4}}){re.escape(suffix)}$")
    indexes = [
        int(match.group(1))
        for path in folder.glob(f"{prefix}_iter_*{suffix}")
        if (match := pattern.match(path.name)) is not None
    ]
    index = max(indexes, default=0) + 1
    return index, folder / f"{prefix}_iter_{index:04d}{suffix}"


def validate_model_spaces(model: Any, env: Any) -> None:
    if model.observation_space != env.observation_space:
        raise ValueError(
            f"续训观测空间不匹配：{model.observation_space} != {env.observation_space}"
        )
    if model.action_space != env.action_space:
        raise ValueError(f"续训动作空间不匹配：{model.action_space} != {env.action_space}")


def run_episode(
    env: Any,
    *,
    seed: int,
    model: Any | None = None,
    maximum_steps: int | None = None,
) -> dict[str, Any]:
    observation, info = env.reset(seed=seed)
    limit = int(maximum_steps if maximum_steps is not None else env.max_steps)
    total_reward = 0.0
    for _ in range(limit):
        if model is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    result = dict(info)
    result["seed"] = int(seed)
    result["episode_reward"] = float(total_reward)
    return result


def evaluate(
    env_factory: Callable[[int], Any],
    seeds: list[int],
    *,
    model: Any | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        env = env_factory(seed)
        try:
            rows.append(run_episode(env, seed=seed, model=model))
        finally:
            env.close()
    return rows


def basic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("评估至少需要一个回合")
    return {
        "episode_count": len(rows),
        "success_count": int(sum(bool(row.get("success")) for row in rows)),
        "success_rate": float(np.mean([bool(row.get("success")) for row in rows])),
        "fall_rate": float(np.mean([bool(row.get("fell")) for row in rows])),
        "finite_state_rate": float(np.mean([bool(row.get("finite_state")) for row in rows])),
        "all_23_targets_sent_rate": float(
            np.mean([bool(row.get("all_23_targets_sent")) for row in rows])
        ),
        "mean_episode_reward": float(np.mean([float(row["episode_reward"]) for row in rows])),
    }
