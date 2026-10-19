# v041 主动抢断

本课程由脚本攻击者带球向右侧球门推进，学习方防守者预测球的短期位置，从球门侧拦截。成功必须出现攻击者到防守者的真实接触换权，并由防守者持续控制；上身撞球、摔倒、NaN或未发送完整23路目标都会失败。

## 当前实测

| 证据 | 结果 |
|---|---|
| `metrics/v041_smoke_8_steps.json` | 8步安全冒烟通过 |
| `metrics/v041_teacher_fixed_eval_run_05.json` | 教师7/10，换权10/10，0摔倒 |
| `metrics/v041_policy_fixed_eval.json` | 512步PPO候选8/10，换权10/10，0摔倒 |
| `metrics/v041_policy_held_out_test.json` | 20个未见种子15/20，75%，0摔倒 |
| `metrics/v041_policy_iter_0002_training.json` | 从 iter_0001 续训256步，总步数768，未覆盖旧模型 |
| `metrics/v041_policy_fixed_eval_run_02.json` | 续训候选回落到7/10，因此仍以 iter_0001 作为较好研究候选 |

上述结果证明高层残差学习能略微改善拦截后的控球，但还不是比赛级主动抢断。当前只有一种直线脚本攻击者，成功判定只要求连续控制0.4秒；正式合同要求至少三类脚本带球者和历史策略池、100个封存种子、夺回率不低于65%，抢断后安全控球或传球至少2秒，并统计犯规、被过和失位。因此 `v041` 仍不得晋级团队比赛。

## 2026-08-20 严格门槛复审

独立复审发现旧成功判定遗漏了配置中的0.25米清球条件，也没有统计双方机器人碰撞。旧固定集8个“成功”全部低于0.25米清球线；物理复现还出现躯干、头部相撞，因此旧 `quality_passed` 和旧75%封存数字不可作为合法抢断证据。

现在成功必须同时满足真实换权、连续控球、至少0.25米清球、无上身碰球、无摔倒，并以保守规则把任何跨队身体接触判为疑似冲撞犯规。新评估 `metrics/v041_policy_fixed_eval_run_03.json` 为0/10，犯规回合率100%，上身碰撞回合率80%，`quality_passed=false`。

```bash
python experiments/v041_active_tackle/scripts/smoke_v041.py
python experiments/v041_active_tackle/scripts/train_v041.py --timesteps 512
python experiments/v041_active_tackle/scripts/train_v041.py --timesteps 256 --resume experiments/v041_active_tackle/checkpoints/v041_policy_iter_0001.zip
python experiments/v041_active_tackle/scripts/evaluate_v041.py --model experiments/v041_active_tackle/checkpoints/v041_policy_iter_0001.zip
python experiments/v041_active_tackle/scripts/evaluate_v041.py --model experiments/v041_active_tackle/checkpoints/v041_policy_iter_0001.zip --held-out
```
