"""Evaluate a v039 checkpoint or the safe zero-action reference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v039_getup_recovery.artifacts import write_json  # noqa: E402
from experiments.v039_getup_recovery.evaluation import evaluate  # noqa: E402
from experiments.v039_getup_recovery.getup_env import load_config  # noqa: E402


FIXED_VALIDATION_SEED_START = 39_100
HELD_OUT_TEST_SEED_START = 1_039_100


def resolve_evaluation_request(
    *,
    held_out: bool,
    seed_start: int | None,
    episodes_per_pose: int | None,
    config: dict[str, object],
) -> tuple[str, int, int]:
    """Choose a real validation/test split and reject overlapping held-out seeds."""

    gate = config["quality_gate"]
    if not isinstance(gate, dict):
        raise ValueError("quality_gate must be an object")
    split = "held_out_test" if held_out else "fixed_validation"
    default_count_key = (
        "held_out_episodes_per_pose" if held_out else "validation_episodes_per_pose"
    )
    count = int(gate[default_count_key]) if episodes_per_pose is None else int(episodes_per_pose)
    if count <= 0:
        raise ValueError("episodes-per-pose must be positive")
    start = (
        HELD_OUT_TEST_SEED_START if held_out else FIXED_VALIDATION_SEED_START
    ) if seed_start is None else int(seed_start)
    if start < 0:
        raise ValueError("seed-start must be non-negative")

    if held_out:
        validation_count = int(gate["validation_episodes_per_pose"])
        fixed_start = FIXED_VALIDATION_SEED_START
        fixed_stop = fixed_start + validation_count
        held_out_stop = start + count
        overlaps_fixed_validation = start < fixed_stop and held_out_stop > fixed_start
        if overlaps_fixed_validation:
            raise ValueError(
                "held-out seeds overlap the reserved fixed-validation range "
                f"[{fixed_start}, {fixed_stop}); choose another --seed-start"
            )
    return split, start, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument(
        "--episodes-per-pose",
        type=int,
        default=None,
        help="defaults to the configured validation count, or held-out count with --held-out",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="defaults to 39100 for validation and 1039100 for the held-out test",
    )
    parser.add_argument("--held-out", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.checkpoint) == bool(args.zero_action):
        raise SystemExit("Choose exactly one of --checkpoint or --zero-action")
    config = load_config()
    try:
        split, seed_start, episodes_per_pose = resolve_evaluation_request(
            held_out=bool(args.held_out),
            seed_start=args.seed_start,
            episodes_per_pose=args.episodes_per_pose,
            config=config,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    policy = None
    if args.checkpoint:
        from stable_baselines3 import PPO

        policy = PPO.load(str(args.checkpoint), device=args.device)
    seeds = list(range(seed_start, seed_start + episodes_per_pose))
    payload = evaluate(
        policy,
        stage=args.stage,
        seeds=seeds,
        zero_action=args.zero_action,
    )
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["checkpoint"] = (
        str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None
    )
    payload["policy"] = "zero_action_prepare_pose" if args.zero_action else "PPO"
    payload["evaluation_split"] = split
    payload["seed_start"] = seed_start
    payload["seed_stop_exclusive"] = seed_start + episodes_per_pose
    suffix = "held_out_test" if args.held_out else "fixed_eval"
    if args.checkpoint:
        base_name = args.checkpoint.stem
    else:
        base_name = f"v039_stage_{args.stage:02d}_zero_action"
    output = write_json(
        Path(__file__).resolve().parents[1] / "metrics" / f"{base_name}_{suffix}.json",
        payload,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
