# v035 7v7 self-play

状态：比赛与训练代码已完成，8 步安全冒烟和完整 10 分钟启发式基线已通过；行为克隆/IPPO 长训练被 v034 质量门槛锁定。

## 已完成验证

- 23 DoF 审计：通过。
- 14 台×23 路、8 步冒烟：`metrics/smoke_0002.json`，14 台全部站立。
- 控制器朝向回归：修正自由关节角速度坐标后，朝向 0 和 π 的两队均稳定。
- 单机 13 秒站立/前进/横移/转向：全部无跌倒，见 `records/v032_yaw_frame_fix_20260718.json`。
- 完整比赛：`matches/match_0007_summary.json`，两半场各 300 秒、下半场换边、14 台全程站立、0:0。
- 演示数据：`match_0007_demonstrations.npz`，84,000 个球员级样本。

## 入口

```bash
# 8 步冒烟
python experiments/v035_7v7_self_play/scripts/smoke_v035.py

# 完整无窗口比赛
python -m experiments.v035_7v7_self_play.match_7v7 --half-seconds 300 --halves 2

# 模型比赛窗口
bash scripts/launch_v035_match_viewer.sh

# 门槛通过后先行为克隆
python -m experiments.v035_7v7_self_play.train_imitation \
  --dataset experiments/v035_7v7_self_play/matches/match_0007_demonstrations.npz

# 再从行为克隆模型继续 IPPO
python -m experiments.v035_7v7_self_play.train_ippo \
  --resume experiments/v035_7v7_self_play/checkpoints/v035_imitation_0001.pt
```

训练器会主动读取 v034 最新指标。`quality_passed=false` 时抛出错误是预期安全行为，不是安装故障。

