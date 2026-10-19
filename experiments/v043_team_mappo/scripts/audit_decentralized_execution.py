"""Audit that the deployed v043 Actor has no centralized state or teammate-row leak."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments.v043_team_mappo.artifacts import next_numbered_path
from experiments.v043_team_mappo.ball_events import BallEventSnapshot
from experiments.v043_team_mappo.networks import RecurrentMAPPO
from experiments.v043_team_mappo.roles import assign_decentralized_roles
from experiments.v043_team_mappo.spec import LOCAL_OBSERVATION_DIM, Role


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


class _ForbiddenCritic(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise AssertionError("central critic was called during decentralized action inference")


def _fake_agent(number: int, x: float, y: float, yaw: float = 0.0) -> object:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return SimpleNamespace(
        xpos=np.asarray([x, y, 0.67], dtype=np.float64),
        xmat=rotation.reshape(-1),
        agent_id=SimpleNamespace(player_no=number),
    )


def main() -> None:
    torch.manual_seed(43004)
    model = RecurrentMAPPO().eval()
    model.central_critic = _ForbiddenCritic()
    observations = torch.randn(3, LOCAL_OBSERVATION_DIM)
    hidden = model.initial_hidden(3)
    with torch.inference_mode():
        baseline_actions, baseline_hidden = model.deterministic_step(observations, hidden)
        changed = observations.clone()
        changed[1:] = torch.randn_like(changed[1:]) * 100.0
        changed_actions, changed_hidden = model.deterministic_step(changed, hidden)
    row_isolation = bool(
        torch.equal(baseline_actions[0], changed_actions[0])
        and torch.equal(baseline_hidden[0], changed_hidden[0])
    )

    global_event = BallEventSnapshot(
        active_team=1,
        active_player=2,
        last_touch_team=1,
        last_touch_player=2,
        touch_started=True,
        turnover=True,
        contact_age_s=0.1,
    )
    touching_player = global_event.encoded_for_local_actor(1, 2, ball_visible=True)
    other_player = global_event.encoded_for_local_actor(1, 3, ball_visible=True)
    event_not_broadcast = bool(
        touching_player[1] == 1.0
        and touching_player[3] == 1.0
        and other_player[1] == 0.0
        and other_player[2] == 0.0
        and other_player[3] == 0.0
        and other_player[4] == 0.0
        and other_player[5] == 0.0
    )

    players = {
        1: _fake_agent(1, -24.5, 0.0),
        2: _fake_agent(2, 0.0, 0.0),
        3: _fake_agent(3, -5.0, 0.0),
    }
    ball = np.asarray([5.0, 0.0], dtype=np.float64)
    roles_before = assign_decentralized_roles(players, ball, side_sign=1)
    players[3] = _fake_agent(3, -15.0, 8.0)
    roles_after = assign_decentralized_roles(players, ball, side_sign=1)
    invisible_teammate_invariance = bool(
        roles_before[2] == roles_after[2]
    )

    forbidden_imports = {"socket", "subprocess", "multiprocessing", "requests", "zmq"}
    source_files = [
        EXPERIMENT_ROOT / "networks.py",
        EXPERIMENT_ROOT / "observations.py",
        EXPERIMENT_ROOT / "roles.py",
    ]
    imported: set[str] = set()
    for source in source_files:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    no_ipc_import = not bool(imported & forbidden_imports)

    passed = row_isolation and event_not_broadcast and invisible_teammate_invariance and no_ipc_import
    output = next_numbered_path(
        EXPERIMENT_ROOT / "metrics", "v043_decentralized_execution_audit", ".json"
    )
    payload = {
        "version": "v043_team_mappo",
        "actor_teammate_row_isolation": row_isolation,
        "central_critic_not_called_by_actor": True,
        "global_ball_event_not_broadcast": event_not_broadcast,
        "invisible_teammate_role_invariance": invisible_teammate_invariance,
        "no_socket_or_ipc_import_in_deployment_path": no_ipc_import,
        "single_process_note": (
            "The simulator batches agents in one process for speed. Each Actor row is independent and has its own "
            "GRU state; this is not an inter-player communication channel."
        ),
        "ctde_note": (
            "Global state, global contacts and privileged teacher roles are used only for training critic, rewards, "
            "teacher baseline and metrics. They are not inputs to deterministic_step."
        ),
        "quality_passed": passed,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(output)}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
