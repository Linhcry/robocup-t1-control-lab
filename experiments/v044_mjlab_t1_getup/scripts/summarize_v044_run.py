"""Create immutable evidence for a completed v044 MJLab training run.

Training scalars are useful for detecting collapse and selecting candidates for
validation, but they are deliberately never converted into a promotion result.
Only the independent RCSS evaluator can set a model quality gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_from_yaml(path: Path, key: str) -> int:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return int(match.group(1))
    raise ValueError(f"could not find integer {key!r} in {path}")


def _scalar_summary(events: list[Any]) -> dict[str, Any]:
    values = np.asarray([float(event.value) for event in events], dtype=np.float64)
    steps = np.asarray([int(event.step) for event in events], dtype=np.int64)
    maximum = int(np.argmax(values))
    minimum = int(np.argmin(values))
    return {
        "count": len(events),
        "first": {"step": int(steps[0]), "value": float(values[0])},
        "last": {"step": int(steps[-1]), "value": float(values[-1])},
        "minimum": {"step": int(steps[minimum]), "value": float(values[minimum])},
        "maximum": {"step": int(steps[maximum]), "value": float(values[maximum])},
        "last_50_mean": float(np.mean(values[-min(50, len(values)) :])),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-iterations", type=int, default=3000)
    args = parser.parse_args()
    if args.expected_iterations <= 0:
        parser.error("expected-iterations must be positive")
    return args


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite historical evidence: {output}")
    agent_yaml = run_dir / "params" / "agent.yaml"
    env_yaml = run_dir / "params" / "env.yaml"
    if not agent_yaml.is_file() or not env_yaml.is_file():
        raise FileNotFoundError("run is missing params/agent.yaml or params/env.yaml")
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise ValueError(f"expected exactly one TensorBoard event file, found {len(event_files)}")

    checkpoints: list[dict[str, Any]] = []
    for path in run_dir.glob("model_*.pt"):
        match = CHECKPOINT_RE.match(path.name)
        if match:
            checkpoints.append(
                {
                    "iteration": int(match.group(1)),
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    checkpoints.sort(key=lambda item: item["iteration"])
    if not checkpoints:
        raise ValueError("run contains no model_N.pt checkpoints")

    accumulator = EventAccumulator(
        str(event_files[0]), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    scalars = {
        tag: _scalar_summary(accumulator.Scalars(tag))
        for tag in scalar_tags
        if accumulator.Scalars(tag)
    }
    metric_tag = "Episode_Metrics/getup_success"
    metric_events = accumulator.Scalars(metric_tag)
    if not metric_events:
        raise ValueError(f"run is missing required scalar {metric_tag}")
    metric_by_step = {
        int(event.step): float(event.value) for event in metric_events
    }
    for checkpoint in checkpoints:
        checkpoint["training_getup_success_at_iteration"] = metric_by_step.get(
            checkpoint["iteration"]
        )

    num_envs = _integer_from_yaml(env_yaml, "num_envs")
    steps_per_env = _integer_from_yaml(agent_yaml, "num_steps_per_env")
    last_scalar_step = max(int(event.step) for event in metric_events)
    completed_iterations = last_scalar_step + 1
    run_complete = completed_iterations >= args.expected_iterations
    payload = {
        "schema_version": "v044_mjlab_training_summary_v1",
        "experiment": "v044_mjlab_t1_getup",
        "run_directory": str(run_dir),
        "expected_iterations": args.expected_iterations,
        "completed_iterations_from_events": completed_iterations,
        "run_complete": run_complete,
        "num_environments": num_envs,
        "steps_per_environment_per_iteration": steps_per_env,
        "total_environment_steps": num_envs * steps_per_env * completed_iterations,
        "agent_config": {"path": str(agent_yaml), "sha256": _sha256(agent_yaml)},
        "environment_config": {"path": str(env_yaml), "sha256": _sha256(env_yaml)},
        "tensorboard_event": {
            "path": str(event_files[0]),
            "sha256": _sha256(event_files[0]),
        },
        "checkpoints": checkpoints,
        "scalar_summaries": scalars,
        "training_metric_is_not_promotion_evidence": True,
        "validation_passed": False,
        "quality_passed": False,
        "reason_quality_is_false": (
            "The upstream getup_success scalar only records whether a robot ever "
            "reached the height/upright condition and includes training-domain "
            "resets. Candidate checkpoints still require strict RCSS four-pose "
            "validation and a disjoint held-out test."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "run_complete": run_complete,
                "completed_iterations": completed_iterations,
                "total_environment_steps": payload["total_environment_steps"],
                "checkpoints": len(checkpoints),
                "quality_passed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if run_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
