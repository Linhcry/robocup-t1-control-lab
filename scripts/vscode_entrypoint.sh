#!/usr/bin/env bash
# VS Code 的所有机器人任务都汇总到这个入口，避免用户记忆长命令。
set -euo pipefail

# 禁止崩溃时生成数百 MB 的 core 文件，并限制数值库线程，避免挤占 VS Code Server。
ulimit -c 0 || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "$(dirname "$0")/.."
source "$HOME/robot_soccer/envs/rl-mujoco/bin/activate"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

usage() {
  echo "用法: bash scripts/vscode_entrypoint.sh {audit|evaluate-walk|record-walk|evaluate-kick|record-kick|metrics|v034-smoke|v034-train-short|v034-evaluate|v034-status|v035-smoke|v035-match-10m|v035-match-viewer|v035-server-viewer|v035-status|v035-train-ippo|v036-train-short|v036-evaluate|v036-record|v036-status|v036-viewer|v037-continue-midfield|v037-evaluate-midfield|v037-record-midfield|v037-status|v037-viewer|v038-gait|v038-smoke|v038-teacher-baseline|v038-bootstrap-baseline|v038-train-stage0|v038-train-stage0-safe|v038-evaluate-stage0|v038-train-stage1-safe|v038-evaluate-stage1|v038-viewer|v038-record|v038-status}" >&2
  exit 2
}

case "${1:-}" in
  audit)
    python scripts/audit_t1_23dof.py
    ;;
  evaluate-walk)
    python locomotion/evaluate_rcss_official_deploy.py \
      --prepare 3 \
      --duration 8 \
      --yaw-rate 0.20 \
      --output records/v032_vscode_free_walk_eval.json
    ;;
  record-walk)
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    python locomotion/record_free_walk.py
    ;;
  evaluate-kick)
    python locomotion/record_walk_kick.py \
      --no-video \
      --metrics records/v033_vscode_walk_kick_eval.json
    ;;
  record-kick)
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    python locomotion/record_walk_kick.py
    ;;
  v034-smoke)
    python experiments/v034_dynamic_single_leg_kick/scripts/smoke_v034.py
    ;;
  v034-train-short)
    python experiments/v034_dynamic_single_leg_kick/scripts/train_v034.py --timesteps 2048
    ;;
  v034-evaluate)
    python experiments/v034_dynamic_single_leg_kick/scripts/evaluate_v034.py
    ;;
  v034-status)
    python - <<'PY'
import json
from pathlib import Path

root = Path("experiments/v034_dynamic_single_leg_kick")
checkpoints = sorted(root.joinpath("checkpoints").glob("v034_policy_iter_*.zip"))
metrics = sorted(root.joinpath("metrics").glob("v034_policy_iter_*_training.json"))
print(f"checkpoint_count={len(checkpoints)}")
print(f"latest_checkpoint={checkpoints[-1] if checkpoints else 'none'}")
if metrics:
    payload = json.loads(metrics[-1].read_text(encoding="utf-8"))
    print(json.dumps({
        "metrics": str(metrics[-1]),
        "total_model_timesteps": payload.get("total_model_timesteps"),
        "quality_passed": payload.get("quality_passed"),
        "summary": payload.get("summary"),
    }, ensure_ascii=False, indent=2))
PY
    ;;
  v035-smoke)
    python experiments/v035_7v7_self_play/scripts/smoke_v035.py
    ;;
  v035-match-10m)
    python -m experiments.v035_7v7_self_play.match_7v7 --half-seconds 300 --halves 2
    ;;
  v035-match-viewer)
    bash scripts/launch_v035_match_viewer.sh
    ;;
  v035-server-viewer)
    bash scripts/launch_rcssservermj_viewer.sh
    ;;
  v035-train-ippo)
    python -m experiments.v035_7v7_self_play.train_ippo --updates 1 --rollout-steps 64
    ;;
  v035-status)
    python - <<'PY'
import json
from pathlib import Path
from experiments.v035_7v7_self_play.gate_status import latest_v034_metrics, v035_training_allowed

root = Path("experiments/v035_7v7_self_play")
allowed, reason = v035_training_allowed()
v034_path, v034_payload = latest_v034_metrics()
print(json.dumps({
    "v035_training_allowed": allowed,
    "reason": reason,
    "v034_latest_metrics": None if v034_path is None else str(v034_path),
    "v034_quality_passed": None if v034_payload is None else v034_payload.get("quality_passed"),
    "v035_smokes": len(list(root.joinpath("metrics").glob("smoke_*.json"))),
    "v035_matches": len(list(root.joinpath("matches").glob("match_*_summary.json"))),
    "v035_checkpoints": len(list(root.joinpath("checkpoints").glob("v035_*.pt"))),
}, ensure_ascii=False, indent=2))
PY
    ;;
  v036-train-short)
    python experiments/v036_single_robot_score/scripts/train_v036.py --timesteps 4096
    ;;
  v036-evaluate)
    python experiments/v036_single_robot_score/scripts/evaluate_v036.py
    ;;
  v036-record)
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    python experiments/v036_single_robot_score/scripts/record_v036.py
    ;;
  v036-status)
    python - <<'PY'
import json
from pathlib import Path

root = Path("experiments/v036_single_robot_score")
checkpoints = sorted(root.joinpath("checkpoints").glob("v036_policy_iter_*.zip"))
metrics = sorted(root.joinpath("metrics").glob("v036_policy_iter_*_training.json"))
payload = None if not metrics else json.loads(metrics[-1].read_text(encoding="utf-8"))
print(json.dumps({
    "checkpoint_count": len(checkpoints),
    "latest_checkpoint": None if not checkpoints else str(checkpoints[-1]),
    "latest_metrics": None if not metrics else str(metrics[-1]),
    "quality_passed": None if payload is None else payload.get("quality_passed"),
    "summary": None if payload is None else payload.get("summary"),
}, ensure_ascii=False, indent=2))
PY
    ;;
  v036-viewer)
    unset MUJOCO_GL PYOPENGL_PLATFORM || true
    python scripts/view_trained_model_live.py --experiment v036
    ;;
  v037-continue-midfield)
    python experiments/v037_midfield_dribble_score/scripts/train_residual_stage.py \
      --stage 4 \
      --timesteps 4096
    ;;
  v037-evaluate-midfield)
    python experiments/v037_midfield_dribble_score/scripts/evaluate_residual_stage.py --stage 4
    ;;
  v037-record-midfield)
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    python experiments/v037_midfield_dribble_score/scripts/record_midfield_goal.py
    ;;
  v037-status)
    python - <<'PY'
import json
from pathlib import Path

root = Path("experiments/v037_midfield_dribble_score")
checkpoints = sorted(root.joinpath("checkpoints").glob("v037r2_stage_04_iter_*.zip"))
metrics = sorted(root.joinpath("metrics").glob("v037r2_stage_04_iter_*_fixed_eval.json"))
payload = None if not metrics else json.loads(metrics[-1].read_text(encoding="utf-8"))
print(json.dumps({
    "latest_checkpoint": None if not checkpoints else str(checkpoints[-1]),
    "latest_independent_evaluation": None if not metrics else str(metrics[-1]),
    "quality_passed": None if payload is None else payload.get("quality_passed"),
    "summary": None if payload is None else payload.get("summary"),
}, ensure_ascii=False, indent=2))
PY
    ;;
  v037-viewer)
    unset MUJOCO_GL PYOPENGL_PLATFORM || true
    python scripts/view_trained_model_live.py --experiment v037
    ;;
  v038-gait)
    python experiments/v038_high_speed_dribble/scripts/benchmark_gait.py
    ;;
  v038-smoke)
    python experiments/v038_high_speed_dribble/scripts/smoke_v038.py
    ;;
  v038-teacher-baseline)
    python experiments/v038_high_speed_dribble/scripts/evaluate_stage.py --stage 0 --teacher-only
    ;;
  v038-bootstrap-baseline)
    python experiments/v038_high_speed_dribble/scripts/evaluate_stage.py \
      --stage 0 \
      --model experiments/v037_midfield_dribble_score/checkpoints/v037r2_stage_04_iter_0002.zip
    ;;
  v038-train-stage0)
    python experiments/v038_high_speed_dribble/scripts/train_stage.py --stage 0 --timesteps 4096
    ;;
  v038-train-stage0-safe)
    python experiments/v038_high_speed_dribble/scripts/train_stage.py \
      --stage 0 \
      --timesteps 4096 \
      --safe-reset-head
    ;;
  v038-evaluate-stage0)
    python experiments/v038_high_speed_dribble/scripts/evaluate_stage.py --stage 0
    ;;
  v038-train-stage1-safe)
    python experiments/v038_high_speed_dribble/scripts/train_stage.py \
      --stage 1 \
      --timesteps 4096 \
      --safe-reset-head
    ;;
  v038-evaluate-stage1)
    python experiments/v038_high_speed_dribble/scripts/evaluate_stage.py --stage 1
    ;;
  v038-viewer)
    unset MUJOCO_GL PYOPENGL_PLATFORM || true
    python scripts/view_trained_model_live.py --experiment v038
    ;;
  v038-record)
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    python experiments/v038_high_speed_dribble/scripts/record_promoted_goal.py
    ;;
  v038-status)
    python - <<'PY'
import json
from pathlib import Path
import re

root = Path("experiments/v038_high_speed_dribble")
pattern = re.compile(r"stage_(\d+)_iter_(\d+)")

def rank(path):
    match = pattern.search(path.name)
    return (-1, -1, path.name) if match is None else (int(match.group(1)), int(match.group(2)), path.name)

checkpoints = sorted(root.joinpath("checkpoints").glob("v038_stage_*_iter_*.zip"), key=rank)
training_metrics = sorted(root.joinpath("metrics").glob("v038_stage_*_iter_*_training*.json"), key=rank)
held_out_metrics = sorted(root.joinpath("metrics").glob("v038_stage_*_iter_*_held_out_test*.json"), key=rank)
training_payload = None if not training_metrics else json.loads(training_metrics[-1].read_text(encoding="utf-8"))
held_out_payloads = [
    (path, json.loads(path.read_text(encoding="utf-8"))) for path in held_out_metrics
]
latest_held_out = None if not held_out_payloads else held_out_payloads[-1]
promoted = [(path, payload) for path, payload in held_out_payloads if payload.get("quality_passed")]
print(json.dumps({
    "latest_candidate_checkpoint": None if not checkpoints else str(checkpoints[-1]),
    "latest_training_metrics": None if not training_metrics else str(training_metrics[-1]),
    "validation_passed": None if training_payload is None else training_payload.get("quality_passed"),
    "validation_summary": None if training_payload is None else training_payload.get("summary"),
    "latest_held_out_metrics": None if latest_held_out is None else str(latest_held_out[0]),
    "latest_held_out_passed": None if latest_held_out is None else latest_held_out[1].get("quality_passed"),
    "latest_held_out_summary": None if latest_held_out is None else latest_held_out[1].get("summary"),
    "promoted_checkpoint": None if not promoted else promoted[-1][1].get("model"),
    "promoted_held_out_metrics": None if not promoted else str(promoted[-1][0]),
}, ensure_ascii=False, indent=2))
PY
    ;;
  metrics)
    python - <<'PY'
import json
from pathlib import Path

for path in [
    Path("records/v032_vscode_free_walk_eval.json"),
    Path("records/v032_rcss_free_walk_video.json"),
    Path("records/v033_vscode_walk_kick_eval.json"),
]:
    print(f"\n===== {path.name} =====")
    if path.exists():
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    else:
        print("尚未同步或生成该指标文件")
PY
    ;;
  *)
    usage
    ;;
esac
