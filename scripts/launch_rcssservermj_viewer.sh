#!/usr/bin/env bash
# 只打开官方 7v7 服务器和内置可视化窗口；此入口不加载我方模型。
set -euo pipefail

cd "$(dirname "$0")/.."
source "$HOME/robot_soccer/envs/rl-mujoco/bin/activate"
exec rcssservermj \
  --field fifa7vs7 \
  --rules ssim26 \
  --render \
  --realtime

