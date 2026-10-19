#!/usr/bin/env bash
# 打开真正加载 T1.pt 和两队战术策略的 10 分钟比赛窗口。
set -euo pipefail

cd "$(dirname "$0")/.."
source "$HOME/robot_soccer/envs/rl-mujoco/bin/activate"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec python -m experiments.v035_7v7_self_play.match_7v7 \
  --half-seconds 300 \
  --halves 2 \
  --viewer \
  --realtime "$@"

