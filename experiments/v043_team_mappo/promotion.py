"""Strict evidence gate separating executable prototypes from promoted team policies."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

from experiments.v043_team_mappo.ball_events import (
    POSSESSION_EVIDENCE_SCHEMA,
    POSSESSION_HOLD_WINDOW_S,
)
from experiments.v043_team_mappo.networks import load_checkpoint


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "v043_3v3_to_7v7.json"
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]


def _latest_passing_communication_audit() -> tuple[Path | None, bool]:
    candidates = sorted(
        (EXPERIMENT_ROOT / "metrics").glob("v043_decentralized_execution_audit_*.json")
    )
    if not candidates:
        return None, False
    latest = candidates[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return latest, bool(payload.get("quality_passed"))


def _checkpoint_evidence(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Verify the evaluated bytes instead of trusting a checkpoint label in JSON."""

    raw_path = str(evaluation.get("checkpoint", "")).strip()
    if not raw_path:
        return {"exists": False, "format_valid": False, "digest_matches": False}
    checkpoint = Path(raw_path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (PROJECT_ROOT / checkpoint).resolve()
    exists = checkpoint.is_file()
    format_valid = False
    metadata_matches = False
    if exists:
        try:
            _, payload = load_checkpoint(checkpoint, device="cpu")
            format_valid = payload.get("format") == "v043_recurrent_mappo_v2_local_roles"
            metadata_matches = bool(
                int(payload.get("players_per_team", -1))
                == int(evaluation.get("players_per_team", -2))
            )
        except Exception:
            format_valid = False
            metadata_matches = False
    actual_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if exists else None
    expected_digest = str(evaluation.get("checkpoint_sha256", "")).lower()
    return {
        "exists": exists,
        "format_valid": format_valid,
        "metadata_matches": metadata_matches,
        "digest_matches": bool(expected_digest and actual_digest == expected_digest),
        "resolved_path": str(checkpoint),
        "actual_sha256": actual_digest,
    }


def _close(left: object, right: object, *, tolerance: float = 1.0e-6) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _recompute_episode_evidence(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Recompute every promotion statistic from per-match records."""

    episodes = evaluation.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return {"records_valid": False, "summary_matches": False, "episode_count": 0}
    required = {
        "seed",
        "high_level_steps",
        "left_score",
        "right_score",
        "learner_possession_ratio",
        "possession_evidence_schema",
        "possession_hold_window_s",
        "possession_sample_steps",
        "learner_contact_possession_steps",
        "opponent_contact_possession_steps",
        "neutral_possession_steps",
        "active_contact_possession_steps",
        "recent_touch_hold_possession_steps",
        "learner_touch_event_count",
        "opponent_touch_event_count",
        "maximum_fallen_players",
        "all_targets_sent",
        "simulated_seconds",
        "true_pass_attempts",
        "true_pass_completions",
        "opponent_id",
    }
    def valid_row(row: object) -> bool:
        if not isinstance(row, dict) or not required.issubset(row):
            return False
        try:
            possession = float(row["learner_possession_ratio"])
            hold_window = float(row["possession_hold_window_s"])
            sample_steps = int(row["possession_sample_steps"])
            learner_steps = int(row["learner_contact_possession_steps"])
            opponent_steps = int(row["opponent_contact_possession_steps"])
            neutral_steps = int(row["neutral_possession_steps"])
            active_steps = int(row["active_contact_possession_steps"])
            hold_steps = int(row["recent_touch_hold_possession_steps"])
            learner_touches = int(row["learner_touch_event_count"])
            opponent_touches = int(row["opponent_touch_event_count"])
            contact_steps = learner_steps + opponent_steps
            expected_possession = learner_steps / contact_steps if contact_steps else 0.0
            duration = float(row["simulated_seconds"])
            sample_period_s = duration / sample_steps if sample_steps > 0 else float("inf")
            maximum_hold_steps_per_touch = (
                math.ceil(POSSESSION_HOLD_WINDOW_S / sample_period_s) + 1
                if math.isfinite(sample_period_s) and sample_period_s > 0.0
                else 0
            )
            attempts = int(row["true_pass_attempts"])
            completions = int(row["true_pass_completions"])
            int(row["seed"])
            left_score = int(row["left_score"])
            right_score = int(row["right_score"])
            fallen = int(row["maximum_fallen_players"])
            return bool(
                math.isfinite(duration)
                and duration > 0.0
                and math.isfinite(possession)
                and 0.0 <= possession <= 1.0
                and row["possession_evidence_schema"] == POSSESSION_EVIDENCE_SCHEMA
                and _close(hold_window, POSSESSION_HOLD_WINDOW_S)
                and sample_steps > 0
                and sample_steps == int(row.get("high_level_steps", -1))
                and min(
                    sample_steps,
                    learner_steps,
                    opponent_steps,
                    neutral_steps,
                    active_steps,
                    hold_steps,
                    learner_touches,
                    opponent_touches,
                ) >= 0
                and learner_steps + opponent_steps + neutral_steps == sample_steps
                and active_steps + hold_steps == contact_steps
                and hold_steps
                <= (learner_touches + opponent_touches) * maximum_hold_steps_per_touch
                and _close(possession, expected_possession)
                and (learner_steps == 0 or learner_touches > 0)
                and (opponent_steps == 0 or opponent_touches > 0)
                and 0 <= completions <= attempts
                and left_score >= 0
                and right_score >= 0
                and fallen >= 0
                and isinstance(row["all_targets_sent"], bool)
                and bool(str(row["opponent_id"]).strip())
            )
        except (TypeError, ValueError, OverflowError):
            return False

    records_valid = all(valid_row(row) for row in episodes)
    if not records_valid:
        return {
            "records_valid": False,
            "summary_matches": False,
            "episode_count": len(episodes),
        }

    seeds = [int(row["seed"]) for row in episodes]
    declared_seeds = evaluation.get("evaluation_seeds")
    try:
        seed_manifest_valid = bool(
            isinstance(declared_seeds, list)
            and [int(seed) for seed in declared_seeds] == seeds
            and len(set(seeds)) == len(seeds)
        )
    except (TypeError, ValueError, OverflowError):
        seed_manifest_valid = False
    wins = sum(int(row["left_score"]) > int(row["right_score"]) for row in episodes)
    draws = sum(int(row["left_score"]) == int(row["right_score"]) for row in episodes)
    losses = len(episodes) - wins - draws
    learner_possession_steps = sum(
        int(row["learner_contact_possession_steps"]) for row in episodes
    )
    opponent_possession_steps = sum(
        int(row["opponent_contact_possession_steps"]) for row in episodes
    )
    possession_steps = learner_possession_steps + opponent_possession_steps
    possession = learner_possession_steps / possession_steps if possession_steps else 0.0
    possession_sample_steps = sum(int(row["possession_sample_steps"]) for row in episodes)
    learner_touch_events = sum(int(row["learner_touch_event_count"]) for row in episodes)
    opponent_touch_events = sum(int(row["opponent_touch_event_count"]) for row in episodes)
    fall_rate = sum(int(row["maximum_fallen_players"]) > 0 for row in episodes) / len(episodes)
    targets_rate = sum(bool(row["all_targets_sent"]) for row in episodes) / len(episodes)
    pass_attempts = sum(int(row["true_pass_attempts"]) for row in episodes)
    pass_completions = sum(int(row["true_pass_completions"]) for row in episodes)
    pass_rate = pass_completions / pass_attempts if pass_attempts else 0.0
    total_seconds = sum(float(row["simulated_seconds"]) for row in episodes)
    goals_conceded_per_10min = (
        600.0 * sum(int(row["right_score"]) for row in episodes) / total_seconds
    )
    minimum_duration = min(float(row["simulated_seconds"]) for row in episodes)
    opponent_ids = {str(row["opponent_id"]) for row in episodes if str(row["opponent_id"])}
    recomputed = {
        "episodes": len(episodes),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / len(episodes),
        "mean_learner_possession_ratio": possession,
        "contact_backed_possession_sample_rate": possession_steps / max(1, possession_sample_steps),
        "learner_touch_event_count": learner_touch_events,
        "opponent_touch_event_count": opponent_touch_events,
        "fall_episode_rate": fall_rate,
        "all_23_targets_sent_rate": targets_rate,
        "true_pass_attempts": pass_attempts,
        "true_pass_completions": pass_completions,
        "true_pass_completion_rate": pass_rate,
        "mean_goals_conceded_per_10min": goals_conceded_per_10min,
        "minimum_match_duration_s": minimum_duration,
        "distinct_opponents": len(opponent_ids),
    }
    supplied = evaluation.get("summary")
    summary_matches = isinstance(supplied, dict) and all(
        key in supplied and _close(supplied[key], value)
        for key, value in recomputed.items()
    )
    return {
        "records_valid": True,
        "possession_evidence_valid": bool(
            evaluation.get("possession_evidence_schema") == POSSESSION_EVIDENCE_SCHEMA
            and _close(
                evaluation.get("possession_hold_window_s"),
                POSSESSION_HOLD_WINDOW_S,
            )
        ),
        "seed_manifest_valid": seed_manifest_valid,
        "summary_matches": summary_matches,
        "episode_count": len(episodes),
        "opponent_ids": sorted(opponent_ids),
        "recomputed_summary": recomputed,
    }


def assess_promotion(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Require every configured held-out, skill, safety and communication condition."""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    environment_config = config["environment"]
    gates = config["promotion_gates_not_yet_run"]
    checkpoint = _checkpoint_evidence(evaluation)
    evidence = _recompute_episode_evidence(evaluation)
    summary = evidence.get("recomputed_summary", {})
    communication_path, communication_passed = _latest_passing_communication_audit()
    results = {
        "checkpoint_exists": bool(checkpoint.get("exists")),
        "checkpoint_format": bool(checkpoint.get("format_valid")),
        "checkpoint_metadata": bool(checkpoint.get("metadata_matches")),
        "checkpoint_digest": bool(checkpoint.get("digest_matches")),
        "episode_records": bool(evidence.get("records_valid")),
        "seed_manifest": bool(evidence.get("seed_manifest_valid")),
        "summary_recomputed": bool(evidence.get("summary_matches")),
        "contact_backed_possession_evidence": bool(
            evidence.get("possession_evidence_valid")
        ),
        "contact_possession_configuration": bool(
            environment_config.get("possession_evidence_schema")
            == POSSESSION_EVIDENCE_SCHEMA
            and _close(
                environment_config.get("possession_hold_window_s"),
                POSSESSION_HOLD_WINDOW_S,
            )
        ),
        "held_out_split": evaluation.get("evaluation_split") == "held_out",
        "held_out_match_count": int(evidence.get("episode_count", 0))
        >= int(gates["held_out_matches"]),
        "ten_minute_matches": float(summary.get("minimum_match_duration_s", 0.0))
        >= float(gates["minimum_match_duration_s"]),
        "win_rate": float(summary.get("win_rate", 0.0)) >= float(gates["minimum_win_rate_vs_heuristic"]),
        "fall_rate": float(summary.get("fall_episode_rate", 1.0)) <= float(gates["maximum_fall_rate"]),
        "all_23_targets_sent": float(summary.get("all_23_targets_sent_rate", 0.0))
        >= float(gates["minimum_all_23_targets_sent_rate"]),
        "possession": bool(evidence.get("possession_evidence_valid"))
        and int(summary.get("learner_touch_event_count", 0)) > 0
        and float(summary.get("mean_learner_possession_ratio", 0.0))
        >= float(gates["minimum_possession_rate"]),
        "true_pass_completion": int(summary.get("true_pass_attempts", 0)) > 0
        and float(summary.get("true_pass_completion_rate", -1.0))
        >= float(gates["minimum_pass_completion_rate"]),
        "goals_conceded": float(summary.get("mean_goals_conceded_per_10min", float("inf")))
        <= float(gates["maximum_goals_conceded_per_10min"]),
        "opponent_diversity": int(summary.get("distinct_opponents", 0))
        >= int(gates["minimum_distinct_opponents"]),
        "validated_team_skill_interface": bool(evaluation.get("validated_team_skill_interface", False)),
        "decentralized_execution_audit": communication_passed,
        "observation_schema_v2": evaluation.get("observation_schema")
        == "v2_decentralized_local_roles_and_own_touch_only",
    }
    return {
        "gate_results": results,
        "quality_passed": bool(all(results.values())),
        "communication_audit": None if communication_path is None else str(communication_path),
        "checkpoint_evidence": checkpoint,
        "episode_evidence": evidence,
        "failed_gates": [name for name, passed in results.items() if not passed],
    }
