"""Regression: a hand-written summary must never pass the v043 promotion gate."""

from __future__ import annotations

from experiments.v043_team_mappo.promotion import assess_promotion


def main() -> None:
    forged = {
        "observation_schema": "v2_decentralized_local_roles_and_own_touch_only",
        "validated_team_skill_interface": True,
        "opponent_policy": "invented-opponent",
        "opponent_pool_members": [f"invented-{index}" for index in range(5)],
        "summary": {
            "episodes": 20,
            "win_rate": 1.0,
            "fall_episode_rate": 0.0,
            "all_23_targets_sent_rate": 1.0,
            "mean_learner_possession_ratio": 1.0,
            "true_pass_completion_rate": 1.0,
            "mean_goals_conceded_per_10min": 0.0,
        },
    }
    result = assess_promotion(forged)
    assert not result["quality_passed"], result
    assert not result["gate_results"]["checkpoint_exists"]
    assert not result["gate_results"]["episode_records"]
    assert not result["gate_results"]["summary_recomputed"]
    assert not result["gate_results"]["contact_backed_possession_evidence"]
    print("v043 forged-summary promotion regression passed")


if __name__ == "__main__":
    main()
