"""Parameter-shared GRU actor and centralized critic for recurrent MAPPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiments.v043_team_mappo.spec import (
    ACTION_DIM,
    CENTRAL_STATE_DIM,
    LOCAL_OBSERVATION_DIM,
    MAX_PLAYERS_PER_TEAM,
    RECURRENT_HIDDEN_DIM,
)


class RecurrentMAPPO(nn.Module):
    """Decentralized recurrent actor plus a training-only centralized critic."""

    def __init__(
        self,
        observation_dim: int = LOCAL_OBSERVATION_DIM,
        state_dim: int = CENTRAL_STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = RECURRENT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.actor_encoder = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
        )
        self.actor_gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.actor_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -1.0, dtype=torch.float32))
        self.central_critic = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, MAX_PLAYERS_PER_TEAM),
        )

    def initial_hidden(self, player_count: int, *, device: torch.device | str | None = None) -> torch.Tensor:
        target_device = next(self.parameters()).device if device is None else torch.device(device)
        return torch.zeros(int(player_count), self.hidden_dim, device=target_device)

    def actor_step(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance independent hidden states for a batch of decentralized actors."""

        encoded = self.actor_encoder(observations)
        next_hidden = self.actor_gru(encoded, hidden)
        means = self.actor_mean(next_hidden)
        log_stds = self.log_std.expand_as(means)
        return means, log_stds, next_hidden

    def actor_unroll(
        self,
        observations: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay a rollout with hidden-state resets and differentiable time recurrence."""

        if observations.ndim != 3:
            raise ValueError("observations must have [time, players, features] shape")
        hidden = initial_hidden
        means: list[torch.Tensor] = []
        log_stds: list[torch.Tensor] = []
        hidden_sequence: list[torch.Tensor] = []
        for time_index in range(observations.shape[0]):
            reset = episode_starts[time_index].to(observations.dtype)
            hidden = hidden * (1.0 - reset)
            mean, log_std, hidden = self.actor_step(observations[time_index], hidden)
            means.append(mean)
            log_stds.append(log_std)
            hidden_sequence.append(hidden)
        return torch.stack(means), torch.stack(log_stds), torch.stack(hidden_sequence)

    def values(self, central_states: torch.Tensor, player_count: int) -> torch.Tensor:
        """Return one centralized value estimate per active player."""

        return self.central_critic(central_states)[..., : int(player_count)]

    def deterministic_step(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means, _, next_hidden = self.actor_step(observations, hidden)
        return torch.tanh(means), next_hidden


def save_checkpoint(
    path: str | Path,
    model: RecurrentMAPPO,
    *,
    update: int,
    players_per_team: int,
    optimizer: torch.optim.Optimizer | None = None,
    evidence_status: str = "training_checkpoint_not_promoted",
) -> None:
    """Save model metadata so a short-run artifact cannot be mistaken for promotion."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "v043_recurrent_mappo_v2_local_roles",
        "observation_dim": LOCAL_OBSERVATION_DIM,
        "central_state_dim": CENTRAL_STATE_DIM,
        "action_dim": ACTION_DIM,
        "hidden_dim": RECURRENT_HIDDEN_DIM,
        "players_per_team": int(players_per_team),
        "update": int(update),
        "evidence_status": evidence_status,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, destination)


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[RecurrentMAPPO, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("format") != "v043_recurrent_mappo_v2_local_roles":
        raise ValueError(f"not a v043 recurrent MAPPO checkpoint: {checkpoint_path}")
    expected = (LOCAL_OBSERVATION_DIM, CENTRAL_STATE_DIM, ACTION_DIM, RECURRENT_HIDDEN_DIM)
    actual = (
        int(payload.get("observation_dim", -1)),
        int(payload.get("central_state_dim", -1)),
        int(payload.get("action_dim", -1)),
        int(payload.get("hidden_dim", -1)),
    )
    if actual != expected:
        raise ValueError(f"checkpoint tensor contract {actual} != {expected}")
    model = RecurrentMAPPO().to(torch.device(device))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
