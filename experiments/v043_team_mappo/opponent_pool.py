"""Disk-backed snapshot league with an explicit heuristic curriculum member."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.v043_team_mappo.artifacts import next_numbered_path
from experiments.v043_team_mappo.core import TeamView
from experiments.v043_team_mappo.networks import RecurrentMAPPO, load_checkpoint, save_checkpoint
from experiments.v043_team_mappo.tactics import privileged_role_heuristic_actions


@dataclass(frozen=True)
class OpponentMember:
    kind: str
    checkpoint: Path | None = None


class OpponentPool:
    """Sample the teacher and historical policies instead of only the latest copy."""

    def __init__(
        self,
        folder: str | Path,
        *,
        teacher_probability: float = 0.40,
        seed: int = 43002,
    ) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.teacher_probability = float(np.clip(teacher_probability, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

    @property
    def snapshots(self) -> list[Path]:
        return sorted(self.folder.glob("opponent_*.pt"))

    def sample(self) -> OpponentMember:
        snapshots = self.snapshots
        if not snapshots or self.rng.random() < self.teacher_probability:
            return OpponentMember(kind="privileged_dynamic_role_heuristic")
        ranks = np.arange(1, len(snapshots) + 1, dtype=np.float64)
        probabilities = ranks / ranks.sum()
        index = int(self.rng.choice(len(snapshots), p=probabilities))
        return OpponentMember(kind="historical_neural_snapshot", checkpoint=snapshots[index])

    def add(self, model: RecurrentMAPPO, *, update: int, players_per_team: int) -> Path:
        destination = next_numbered_path(self.folder, "opponent", ".pt")
        save_checkpoint(
            destination,
            model,
            update=update,
            players_per_team=players_per_team,
            evidence_status="opponent_pool_snapshot_not_promoted",
        )
        return destination


class OpponentController:
    """Keep independent recurrent state for one sampled opponent."""

    def __init__(self, member: OpponentMember, player_count: int, *, device: str = "cpu") -> None:
        self.member = member
        self.player_count = int(player_count)
        self.device = torch.device(device)
        self.model: RecurrentMAPPO | None = None
        self.hidden: torch.Tensor | None = None
        if member.checkpoint is not None:
            self.model, _ = load_checkpoint(member.checkpoint, device=self.device)
            self.hidden = self.model.initial_hidden(self.player_count, device=self.device)

    def reset(self) -> None:
        if self.model is not None:
            self.hidden = self.model.initial_hidden(self.player_count, device=self.device)

    def actions(self, env: object, view: TeamView) -> np.ndarray:
        if self.model is None:
            return privileged_role_heuristic_actions(
                env.core.simulation,
                env.core.right_players,
                env.core.right_teacher_roles,
                side_sign=-1,
            )
        assert self.hidden is not None
        with torch.inference_mode():
            observation_tensor = torch.from_numpy(view.observations).to(self.device)
            actions, self.hidden = self.model.deterministic_step(observation_tensor, self.hidden)
        return actions.cpu().numpy().astype(np.float32)
