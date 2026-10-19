# v034：动态单腿射门

状态：训练环境、奖励函数、8 步 smoke test、PPO 短训练和固定种子评估已实现。已完成 4 个不覆盖历史的短训练段，共 8192 timesteps；固定种子跌倒率仍为 1.0，因此没有启动长训练，也不能部署到比赛代理。

## 基线

- v032：稳定的官方 Booster T1 速度条件步态。
- v033：闭环走向球并普通触球。
- T1 始终为 23 DoF，关节限位以 `locomotion/t1_spec.py` 为准。

## 目标

- 在单脚支撑期间保持躯干高度和姿态。
- 提高球速并约束射门方向。
- 接触球后恢复稳定站立或行走。
- 拒绝以跌倒、越限或数值异常换取球速。

## 开始长训练前

1. 冻结观测、动作、奖励、终止条件与通过门槛。
2. 运行 23 DoF 审计。
3. 实现并通过最小 smoke test。
4. 只运行短训练，确认无 NaN/Inf、无越限、无异常磁盘增长。
5. 固定种子评估通过后再增加训练时长。

## 当前训练入口

```bash
python experiments/v034_dynamic_single_leg_kick/scripts/smoke_v034.py
python experiments/v034_dynamic_single_leg_kick/scripts/train_v034.py --timesteps 2048
python experiments/v034_dynamic_single_leg_kick/scripts/evaluate_v034.py
```

当前配置为 `configs/v034_residual_ppo_r3.json`。`scripts/launch_v034_nohup.sh` 仅为门槛通过后的长段入口；在 `quality_passed=true` 前不要运行。

## 2026-07-18 固定评估结论

- 全部检查点均保持有限值并发送完整 23 路关节目标。
- v034_policy_iter_0002 的固定种子触球率达到 100%，但仍全部跌倒。
- v034_policy_iter_0003 的最小直立度最高，约为 0.366，但重复触球增多。
- v034_policy_iter_0004 修正了重复触球奖励并收窄探索，方向误差约 5.3°，稳定性没有通过门槛。
- 下一轮应先重做支撑/收腿课程或引入已验证站立恢复策略，不能直接延长训练。
