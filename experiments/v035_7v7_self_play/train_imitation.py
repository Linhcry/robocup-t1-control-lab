"""用 7v7 启发式比赛数据对共享 Actor 做监督行为克隆。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.v035_7v7_self_play.artifacts import next_numbered_path
from experiments.v035_7v7_self_play.gate_status import require_v035_training_gate
from experiments.v035_7v7_self_play.local_tactics import ACTION_DIM, LOCAL_OBS_DIM
from experiments.v035_7v7_self_play.networks import (
    SharedActorCritic,
    load_tactical_checkpoint,
    save_tactical_checkpoint,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    """加载 NPZ、切分训练/验证集、优化 MSE，并保存不覆盖的模型与指标。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    require_v035_training_gate()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    archive = np.load(args.dataset)
    observations = np.asarray(archive["observations"], dtype=np.float32).reshape(-1, LOCAL_OBS_DIM)
    actions = np.asarray(archive["actions"], dtype=np.float32).reshape(-1, ACTION_DIM)
    if len(observations) != len(actions) or len(observations) < 32:
        raise ValueError("演示数据至少需要 32 对 64维观测/3维动作")

    permutation = np.random.permutation(len(observations))
    split = max(1, int(0.9 * len(permutation)))
    train_indexes, validation_indexes = permutation[:split], permutation[split:]
    train_dataset = TensorDataset(
        torch.from_numpy(observations[train_indexes]),
        torch.from_numpy(actions[train_indexes]),
    )
    validation_observations = torch.from_numpy(observations[validation_indexes])
    validation_actions = torch.from_numpy(actions[validation_indexes])
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = SharedActorCritic() if args.resume is None else load_tactical_checkpoint(args.resume)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    loss_function = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        batch_losses: list[float] = []
        for batch_observations, batch_actions in loader:
            predicted_actions = model.deterministic_actions(batch_observations)
            loss = loss_function(predicted_actions, batch_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            validation_prediction = model.deterministic_actions(validation_observations)
            validation_loss = float(loss_function(validation_prediction, validation_actions))
        model.train()
        history.append(
            {
                "epoch": epoch,
                "training_mse": float(np.mean(batch_losses)),
                "validation_mse": validation_loss,
            }
        )

    checkpoint = next_numbered_path(EXPERIMENT_ROOT / "checkpoints", "v035_imitation", ".pt")
    metrics = next_numbered_path(EXPERIMENT_ROOT / "metrics", "v035_imitation", ".json")
    save_tactical_checkpoint(checkpoint, model, training_method="behavior_cloning", update=args.epochs)
    payload = {
        "dataset": str(args.dataset.resolve()),
        "samples": len(observations),
        "epochs": args.epochs,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(metrics)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

