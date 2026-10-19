# v038 高速控球与泛化修复

v038 从 `v037r2_stage_04_iter_0002.zip` 继续，但不把 v037 的 4/5 固定种子结果误当成已经泛化。新增协议把训练随机流、10 个验证种子和 20 个最终封存测试种子分开，并始终记录“纯几何教师零残差”和 PPO 残差的消融结果。

本版本只解决两件事：第一，在不跌倒、不丢球的前提下把速度指令从 0.45 m/s 逐级扩展到 0.60、0.80、1.00 m/s；第二，抑制 PPO 残差破坏可靠教师动作的过拟合。它还不是 3v3/7v7 战术模型，也不把高速行走称作真正跑步。

## 为什么暂时不能叫跑步

把足球移出测试区后重新实测，官方 `T1.pt` 在 1.00 m/s 前进指令下可保持站立，8秒测试的平均机身前向速度为 0.736 m/s，但逐个 20 ms 物理周期检查时双脚腾空比例为 0%。严格定义的跑步应存在腾空相，因此当前是稳定高速行走。真正跑步需要另训底层腿部策略，并同时验收腾空相、足端打滑、能耗、组合转向和受扰恢复。

## 三阶段课程

| 阶段 | 最大前进指令 | 场景 | 升级条件 |
|---|---:|---|---|
| 0 | 0.60 m/s | 7–9 m 短距离控球 | 10 个验证种子成功率≥70%、零跌倒、零丢球、受控速度≥0.24 m/s |
| 1 | 0.80 m/s | 14–16 m 进攻三区 | 上一阶段通过后才允许训练 |
| 2 | 1.00 m/s | 26–27 m 中场 | 上一阶段通过后才允许训练 |

每一级通过验证后才可打开一次 20 种子的 held-out test；测试不用于反复调参。

## 2026-08-19 实验结果

| 对象 | 10种子验证 | 20种子封存测试 | 结论 |
|---|---|---|---|
| 第0阶段安全初始化 PPO | 9/10进球、0摔倒、0丢球、78.22%持球、0.308 m/s | 15/20进球、0摔倒、0丢球、85.17%持球、0.267 m/s | 全部门槛通过，正式晋级 |
| 第1阶段纯教师 | 8/10进球、0摔倒、0丢球、81.43%持球、0.313 m/s | 未开启 | 作为可解释基线 |
| 第1阶段安全初始化 PPO | 9/10进球、0摔倒、0丢球、79.82%持球、0.344 m/s | 15/20进球、0摔倒、1次丢球、79.62%持球、0.347 m/s | 严格门槛失败，不晋级第2阶段 |

第1阶段候选模型是 `v038_stage_01_iter_0002.zip`，但实时展示默认加载最后一个真正晋级的 `v038_stage_00_iter_0002.zip`。封存测试已经开启过，不能再根据那20个种子修改策略并声称它们仍是未见测试；下一轮应使用新的训练/验证种子改进丢球恢复，并预先登记一组全新的最终测试种子。

已晋级模型的23.6秒完整进球展示见 [videos/v038_promoted_goal.mp4](videos/v038_promoted_goal.mp4)，同名 JSON 记录了模型、晋级证据、种子和逐回合指标。

## 运行顺序

```bash
python experiments/v038_high_speed_dribble/scripts/benchmark_gait.py
python experiments/v038_high_speed_dribble/scripts/smoke_v038.py

# 同一验证集先记录两个基线
python experiments/v038_high_speed_dribble/scripts/evaluate_stage.py --stage 0 --teacher-only
python experiments/v038_high_speed_dribble/scripts/evaluate_stage.py --stage 0 \
  --model experiments/v037_midfield_dribble_score/checkpoints/v037r2_stage_04_iter_0002.zip

# 从 v037 读取特征层，并把可能有害的残差输出头归零后训练
python experiments/v038_high_speed_dribble/scripts/train_stage.py \
  --stage 0 --timesteps 4096 --safe-reset-head

# 验证通过后才执行一次最终封存测试
python experiments/v038_high_speed_dribble/scripts/evaluate_stage.py --stage 0 --held-out-test

# 第0阶段封存测试通过后，才允许训练和验证第1阶段
python experiments/v038_high_speed_dribble/scripts/train_stage.py \
  --stage 1 --timesteps 4096 --safe-reset-head
python experiments/v038_high_speed_dribble/scripts/evaluate_stage.py --stage 1

# 展示和录像只会选择 held-out test 真正通过的模型
python experiments/v038_high_speed_dribble/scripts/record_promoted_goal.py
```

所有 checkpoint 和 JSON 都按编号新增，不覆盖 v037 或较早的实验记录。
