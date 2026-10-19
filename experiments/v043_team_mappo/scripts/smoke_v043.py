"""Run the mandatory eight-step 3v3 physics and recurrent-network smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.v043_team_mappo.artifacts import next_numbered_path
from experiments.v043_team_mappo.environment import TeamSelfPlayEnv
from experiments.v043_team_mappo.gate_status import team_training_gate
from experiments.v043_team_mappo.networks import RecurrentMAPPO
from experiments.v043_team_mappo.spec import (
    CENTRAL_STATE_DIM,
    ENTITY_MASK_DIM,
    LOCAL_OBSERVATION_DIM,
    Role,
)
from experiments.v043_team_mappo.tactics import privileged_role_heuristic_actions
from experiments.v043_team_mappo.train_mappo import DEFAULT_LOW_LEVEL_MODEL


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players-per-team", type=int, choices=range(3, 8), default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--prepare-seconds", type=float, default=0.10)
    parser.add_argument("--low-level-model", type=Path, default=DEFAULT_LOW_LEVEL_MODEL)
    args = parser.parse_args()

    prerequisite_passed, prerequisite_reason, _ = team_training_gate()
    env = TeamSelfPlayEnv(
        args.low_level_model,
        players_per_team=args.players_per_team,
        action_repeat=1,
        episode_high_level_steps=max(args.steps + 1, 16),
        prepare_seconds=args.prepare_seconds,
        seed=43003,
    )
    left, right, _ = env.reset()
    model = RecurrentMAPPO().eval()
    left_hidden = model.initial_hidden(args.players_per_team)
    role_coverage: set[int] = set(left.roles) | set(right.roles)
    infos: list[dict[str, object]] = []
    network_finite = True
    with torch.inference_mode():
        for _ in range(args.steps):
            means, log_stds, left_hidden = model.actor_step(
                torch.from_numpy(left.observations), left_hidden
            )
            values = model.values(
                torch.from_numpy(left.central_state).unsqueeze(0), args.players_per_team
            )
            network_finite = network_finite and bool(
                torch.isfinite(means).all()
                and torch.isfinite(log_stds).all()
                and torch.isfinite(values).all()
                and torch.isfinite(left_hidden).all()
            )
            left_actions = privileged_role_heuristic_actions(
                env.core.simulation,
                env.core.left_players,
                env.core.left_teacher_roles,
                side_sign=1,
            )
            right_actions = privileged_role_heuristic_actions(
                env.core.simulation,
                env.core.right_players,
                env.core.right_teacher_roles,
                side_sign=-1,
            )
            left, right, _, done, info = env.step(left_actions, right_actions)
            infos.append(info)
            role_coverage.update(left.roles)
            role_coverage.update(right.roles)
            if done:
                raise RuntimeError("smoke episode terminated before all requested steps")

    safety = env.core.safety_snapshot()
    shapes_passed = bool(
        left.observations.shape == (args.players_per_team, LOCAL_OBSERVATION_DIM)
        and left.central_state.shape == (CENTRAL_STATE_DIM,)
        and left.entity_masks.shape == (args.players_per_team, ENTITY_MASK_DIM)
    )
    masks_passed = bool(
        np.isfinite(left.entity_masks).all()
        and np.all((left.entity_masks == 0.0) | (left.entity_masks == 1.0))
    )
    roles_passed = bool(
        Role.GOALKEEPER in {Role(value) for value in role_coverage}
        and Role.PRESSER in {Role(value) for value in role_coverage}
        and bool({Role.SUPPORTER, Role.COVER} & {Role(value) for value in role_coverage})
        and all(0 <= value < len(Role) for value in role_coverage)
    )
    quality_passed = bool(
        prerequisite_passed
        and network_finite
        and shapes_passed
        and masks_passed
        and roles_passed
        and safety["finite_qpos"]
        and safety["finite_qvel"]
        and safety["finite_ctrl"]
        and safety["all_targets_sent"]
        and safety["standing_players"] == safety["total_players"]
    )
    output = next_numbered_path(
        EXPERIMENT_ROOT / "metrics",
        f"v043_{args.players_per_team}v{args.players_per_team}_smoke",
        ".json",
    )
    payload = {
        "version": "v043_team_mappo",
        "test": f"parameterized_{args.players_per_team}v{args.players_per_team}_eight_step_smoke",
        "players_per_team": args.players_per_team,
        "physics_steps": args.steps,
        "executed_policy": "privileged_dynamic_role_heuristic",
        "neural_network_status": "untrained_forward_pass_only",
        "prerequisite_passed": prerequisite_passed,
        "prerequisite_reason": prerequisite_reason,
        "network_finite": network_finite,
        "shapes_passed": shapes_passed,
        "entity_masks_passed": masks_passed,
        "role_contract_passed": roles_passed,
        "role_ids_seen": sorted(role_coverage),
        "safety": safety,
        "quality_passed": quality_passed,
        "claim_boundary": "This smoke proves wiring and safety only. It is not evidence of learned teamwork or match competitiveness.",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "metrics": str(output)}, ensure_ascii=False, indent=2))
    if not quality_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
