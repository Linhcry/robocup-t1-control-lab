"""7v7 上层深度网络：七名球员共享参数，但分别用自己的局部观测推理。"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn

from experiments.v035_7v7_self_play.local_tactics import ACTION_DIM, LOCAL_OBS_DIM


HIDDEN_DIM: Final[int] = 256


class SharedActorCritic(nn.Module):
    """参数共享 IPPO 网络；Actor 和 Critic 都不读取其他球员的私有观测。"""

    def __init__(self, observation_dim: int = LOCAL_OBS_DIM, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.critic = nn.Linear(HIDDEN_DIM, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回动作均值、广播后的对数标准差和状态价值。"""

        features = self.encoder(observations)
        means = self.actor_mean(features)
        values = self.critic(features).squeeze(-1)
        log_stds = self.log_std.expand_as(means)
        return means, log_stds, values

    def deterministic_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """比赛部署只取 Actor 均值，不采样随机动作。"""

        features = self.encoder(observations)
        return torch.tanh(self.actor_mean(features))


def save_tactical_checkpoint(
    path: Path,
    model: SharedActorCritic,
    *,
    training_method: str,
    update: int,
) -> None:
    """用带元数据的 PyTorch 字典保存模型，避免与底层 ``T1.pt`` 混淆。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "v035_shared_local_actor_critic",
            "observation_dim": LOCAL_OBS_DIM,
            "action_dim": ACTION_DIM,
            "training_method": training_method,
            "update": int(update),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_tactical_checkpoint(path: str | Path, device: str = "cpu") -> SharedActorCritic:
    """加载 v035 上层网络并检查输入输出契约。"""

    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("format") != "v035_shared_local_actor_critic":
        raise ValueError(f"不是 v035 战术模型：{checkpoint_path}")
    if int(payload.get("observation_dim", -1)) != LOCAL_OBS_DIM:
        raise ValueError("战术模型观测维度与当前 64 维局部视觉不一致")
    model = SharedActorCritic().to(torch.device(device))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


class TacticalPolicy:
    """把启发式教师或已训练共享 Actor 统一成 ``predict`` 接口。"""

    def __init__(self, model_path: str | Path | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = None if model_path is None else load_tactical_checkpoint(model_path, device)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """输入形状 ``(7,64)``，返回七名球员各自的三维动作。"""

        from experiments.v035_7v7_self_play.local_tactics import heuristic_local_action

        obs = np.asarray(observations, dtype=np.float32)
        if obs.shape != (7, LOCAL_OBS_DIM) or not np.isfinite(obs).all():
            raise ValueError(f"球队观测应为 (7,{LOCAL_OBS_DIM})，实际为 {obs.shape}")
        if self.model is None:
            return np.stack([heuristic_local_action(row) for row in obs])
        with torch.inference_mode():
            tensor = torch.from_numpy(obs).to(self.device)
            actions = self.model.deterministic_actions(tensor).cpu().numpy()
        return np.clip(actions, -1.0, 1.0).astype(np.float32)
