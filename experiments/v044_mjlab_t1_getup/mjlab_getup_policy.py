"""Deterministic loader for an official mjlab Booster T1 getup actor."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn


OBSERVATION_DIM = 75
ACTION_DIM = 23

# Official mjlab HOME_KEYFRAME in canonical T1 joint order.
HOME_QPOS = np.asarray(
    [
        0.0, 0.0,
        0.0, -1.4, 0.0, -0.4,
        0.0, 1.4, 0.0, 0.4,
        0.0,
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    ],
    dtype=np.float32,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MjlabGetupPolicy:
    """Load the actor mean network without importing the training framework."""

    # The official MJLab/RSL-RL T1 get-up configuration uses clip_actions=None.
    # The downstream RCSS controller clamps the resulting absolute joint target
    # to the mechanical limits, so raw actor actions must remain unchanged here.
    raw_action_clip = None

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.device = torch.device(device)
        self.network = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_DIM),
        ).to(self.device)

        payload = torch.load(
            self.checkpoint, map_location=self.device, weights_only=False
        )
        if not isinstance(payload, dict) or "actor_state_dict" not in payload:
            raise ValueError("checkpoint has no actor_state_dict")
        actor_state = payload["actor_state_dict"]
        expected = {
            "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias",
            "mlp.4.weight", "mlp.4.bias", "mlp.6.weight", "mlp.6.bias",
        }
        missing = expected.difference(actor_state)
        if missing:
            raise ValueError(f"actor checkpoint is missing keys: {sorted(missing)}")
        stripped = {
            key.removeprefix("mlp."): value
            for key, value in actor_state.items()
            if key.startswith("mlp.")
        }
        self.network.load_state_dict(stripped, strict=True)
        self.network.eval()
        self.iteration = int(payload.get("iter", -1))
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)

    def reset(self) -> None:
        self.last_action.fill(0.0)

    def infer(
        self,
        *,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> np.ndarray:
        fields = (
            np.asarray(base_ang_vel, dtype=np.float32),
            np.asarray(projected_gravity, dtype=np.float32),
            np.asarray(joint_pos, dtype=np.float32) - HOME_QPOS,
            np.asarray(joint_vel, dtype=np.float32),
            self.last_action,
        )
        observation = np.concatenate(fields)
        if observation.shape != (OBSERVATION_DIM,):
            raise ValueError(f"expected 75D observation, got {observation.shape}")
        if not np.isfinite(observation).all():
            raise FloatingPointError("non-finite getup observation")
        tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            action = self.network(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise FloatingPointError("getup actor returned an invalid action")
        self.last_action[:] = action
        return action.copy()
