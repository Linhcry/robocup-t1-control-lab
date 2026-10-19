#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source "$HOME/robot_soccer/envs/rl-mujoco/bin/activate"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
ulimit -c 0 || true

experiment="experiments/v034_dynamic_single_leg_kick"
mkdir -p "$experiment/logs"
stamp="$(date +%Y%m%d_%H%M%S)"
log="$experiment/logs/v034_train_${stamp}.log"
pid_file="$experiment/logs/v034_train_${stamp}.pid"
timesteps="${1:-16384}"

nohup python "$experiment/scripts/train_v034.py" --timesteps "$timesteps" >"$log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pid_file"
printf 'pid=%s\nlog=%s\npid_file=%s\n' "$pid" "$log" "$pid_file"

