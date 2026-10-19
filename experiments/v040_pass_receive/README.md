# v040 传球与接球

本目录把“同队球员先后碰到球”升级成可复核的物理事件课程。`BallEventTracker` 直接读取 RCSSServerMJ 的 `active_contact`、`last_contact`、`contact_change` 和接触身体部位，只有出现传球者触球、释放、指定接球者触球并继续控球，才记录一次换权。

## 已实现

- `pass_receive_env.py`：两台 T1 的 2v0 直线短传环境。PPO 输出六维残差，两名球员的所有动作仍经过官方 `T1.pt` 和 23DoF 限位控制器。
- `ball_event_tracker.py`：可供抢断、门将和团队环境复用的原生球权事件跟踪器。
- `skill_core.py`：少量 T1 的安全装配、准备姿态、速度动作和 23DoF 审计。
- `scripts/smoke_v040.py`、`evaluate_v040.py`、`train_v040.py`：冒烟、固定评估、封存评估和 PPO 短训练入口。

## 当前实测

| 证据 | 结果 |
|---|---|
| `metrics/v040_smoke_8_steps.json` | 8步状态有限、0摔倒、两台均发送23路目标 |
| `metrics/v040_teacher_fixed_eval_run_07.json` | 当前窄课程教师10/10，真实换权10/10 |
| `metrics/v040_policy_fixed_eval_run_02.json` | 512步PPO候选9/10，0摔倒 |
| `metrics/v040_policy_held_out_test.json` | 20个未见种子15/20，成功率75%，0摔倒，未通过80%本地门槛 |
| `metrics/v040_policy_iter_0002_training.json` | 从 iter_0001 续训256步，总步数768，保存为新文件 |
| `metrics/v040_policy_fixed_eval_run_03.json` | 续训候选退化到0/10，不得用“最新版”替代较好旧候选 |

## 2026-08-20 严格门槛复审

早期 `fixed_eval`/`held_out_test` 把“接球者是最近触球者且距离球不超过0.9米”累计成控球时间，即使球已经释放也会继续计时；旧的 `quality_passed` 因此不可再作为晋级证据。现在控球计时要求每个连续物理帧都同时满足：接球者当前确有脚、踝、小腿或胫骨触球，人与球距离不超过0.45米，球速不超过1.0 m/s。接触者不变但接触部位由脚变成手/上身时也会立即记为违规。

按新门槛复评同一个 `iter_0001`，`metrics/v040_policy_fixed_eval_run_04.json` 为0/10、`quality_passed=false`。这证明旧模型完成了真实的传球者→接球者触球转换，但没有证明连续0.4秒的受控接球。

## 不能晋级的原因

这一关只是约1.5米的“行走推球式短传”，不是已经学会动态踢球。项目现有 v034 动态单腿踢球训练到8192步后，固定评估仍是0%成功、100%摔倒，平均峰值球速约0.43 m/s、平均前进约0.15 m。因此 v040 目前没有一个稳定的、可控制力度的底层传球执行器。

正式验收要求3–12米、多角度、100个封存种子不低于90%，移动接球后连续控球至少2秒，并加入对手截球风险。本目录当前20种子只有75%，距离和场景范围也不够，`v040` 明确保持“研究候选、不得接入比赛晋级”。

## 运行

```bash
python experiments/v040_pass_receive/scripts/smoke_v040.py
python experiments/v040_pass_receive/scripts/train_v040.py --timesteps 512
python experiments/v040_pass_receive/scripts/train_v040.py --timesteps 256 --resume experiments/v040_pass_receive/checkpoints/v040_policy_iter_0001.zip
python experiments/v040_pass_receive/scripts/evaluate_v040.py --model experiments/v040_pass_receive/checkpoints/v040_policy_iter_0001.zip
python experiments/v040_pass_receive/scripts/evaluate_v040.py --model experiments/v040_pass_receive/checkpoints/v040_policy_iter_0001.zip --held-out
```
