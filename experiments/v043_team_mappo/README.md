# v043：3v3→7v7 循环 MAPPO 团队决策

本版本是独立的新实验，不修改或覆盖 v035。目标是先把 3v3 团队学习闭环建立正确，再让同一套固定维度网络扩展到 7v7。

## 已经真实完成的部分

- 每队人数可设为 3～7；局部观测和集中状态统一做七人上限填充。
- 每名球员只向共享 Actor 提供 92 维局部观测；不可见/不存在的球员槽位有明确 mask。
- 每名球员有独立的 128 维 GRU 隐状态，部署时不需要集中状态。
- 集中式 Critic 使用 96 维全局训练状态，并输出最多七名球员的价值。
- 动态角色包括门将、压迫、接应和协防；比赛 Actor 只根据自己的局部视野推断角色。全局真值角色仅供训练教师使用。
- `BallEventTracker` 直接使用 RCSS 的 `active_contact`、`last_contact` 和 `contact_change`，记录触球、同队换人触球和球权转换。
- 对手池同时包含启发式教师和多个历史神经网络快照，不再只对最新自己训练。
- 3v3 与 7v7 的8步物理冒烟均通过；3v3完成两次极短 MAPPO 更新，第二次确实加载了历史神经网络对手。

## 当前不能宣称的能力

当前 v043 动作仍只有行走速度三维量。它没有调用经过验收的传球、扑救或起身技能，因此不能把“同队球员先后碰球”直接称为成功传球，也不能把门将沿门线移动称为学会扑救。两个短训练检查点各只包含8个环境步，用途是证明训练链可运行，`quality_passed` 被明确写成 `false`，不能用于比赛晋级。

## 证据

完整结果、失败历史和逐条复现命令见 `RESULTS.md`。

- `metrics/v043_3v3_smoke_0003.json`：通信修复后，6台机器人、138路目标、全部站立。
- `metrics/v043_7v7_smoke_0002.json`：通信修复后，14台机器人、322路目标、全部站立。
- `metrics/v043_mappo_local_v2_training_0001.json`：通信审计后，启发式对手下的第1次短更新。
- `metrics/v043_mappo_local_v2_training_0002.json`：通信审计后，历史神经网络快照对手下的第2次短更新。
- `metrics/v043_decentralized_execution_audit_0002.json`：Actor 行隔离、Critic 隔离、本地事件和无 IPC 审计。
- `metrics/v043_fixed_eval_0003.json`：v2 神经网络两局均为0:0且球推进为0，明确未达标。
- `metrics/v043_promotion_gate_0001.json`：严格门槛按预期拦截，未通过比赛数、胜率、真实传球、失球、对手多样性和技能接口六项。
- `checkpoints/v043_mappo_local_v2_0002.pt`：本地角色观测 v2 训练链检查点，状态为“未晋级”。

早期 `v043_mappo_0001.pt`/`0002.pt` 使用了全局角色提示，已由通信审计判定不适合分散比赛部署；文件仅作为失败历史保留，v2 加载器会拒绝该格式。

## 2026-08-20 晋级证据完整性修复

旧晋级函数只读取JSON中的汇总字段，手写一个不存在检查点、也没有逐场记录的“完美汇总”即可误判通过；因此 `v043_promotion_gate_0001.json` 只能证明当时输入失败，不能证明门禁本身防伪。新门禁会重新读取并校验检查点格式、人数元数据和SHA-256，从逐场记录重算胜负、摔倒、23路目标、控球、真实传球、失球率和对手数量，并要求唯一封存种子、20场、每场600秒。缺少任一原始证据即失败。

`metrics/v043_fixed_eval_0004.json` 与 `v043_promotion_gate_0003.json` 是新格式的一步轻量回归：检查点、哈希、逐场记录、种子和重算汇总均通过完整性检查，但比赛数、600秒时长、胜率、真实传球、对手多样性和技能接口仍失败，所以 `quality_passed=false`。`scripts/test_promotion_integrity.py` 另行保证伪造汇总不能晋级。

## 2026-08-20 真实触球控球率修复

`v043_fixed_eval_0001.json` 至 `0004.json` 的控球率仍可能在最近一次真实触球超过1.5秒后退回“离球更近的一方”，因此这些旧文件不能再作为控球率证据。现在 `roles.py` 已完全删除最近距离回退：只有当前 RCSS `active_contact`，或距最近真实触球不超过1.5秒的有限保持窗口，才能标记球权；超过窗口立即记为中立。持续接触期间会逐物理帧刷新时间，重建跟踪器时也保留 RCSS 原始旧时间戳，不能把历史 `last_contact` 伪装成刚发生的触球。

新评测会逐局写入真实触球事件数、当前接触步数、有限保持步数和中立步数。晋级器会重新校验这些计数、比例、证据版本与1.5秒窗口，并限制一次触球能够支持的保持采样数；没有真实触球事件时，即使手写控球率也不能通过。`metrics/v043_fixed_eval_0005.json` 是0.1秒轻量回归，唯一采样步没有触球，所以控球率为0；最终门禁证据 `metrics/v043_promotion_gate_0005.json` 的证据结构和配置校验通过，但控球门和其余正式比赛门仍失败，`quality_passed=false`。边界与伪造计数回归见 `scripts/test_contact_possession.py`。

## WSL 命令

```bash
cd /home/linh/robot_soccer/training_runs/rcss_t1_training_20260706
source ~/robot_soccer/envs/rl-mujoco/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# 先执行安全审计与3v3冒烟
python scripts/audit_t1_23dof.py
python experiments/v043_team_mappo/scripts/audit_decentralized_execution.py
python experiments/v043_team_mappo/scripts/smoke_v043.py --players-per-team 3

# 训练链短测试；不是正式训练成绩
python -m experiments.v043_team_mappo.train_mappo \
  --players-per-team 3 --updates 1 --rollout-steps 8 --epochs 1

# 固定种子评估神经网络对启发式教师
python -m experiments.v043_team_mappo.evaluate \
  --checkpoint experiments/v043_team_mappo/checkpoints/v043_mappo_local_v2_0002.pt \
  --players-per-team 3 --episodes 5

# 对评估逐项执行严格晋级门槛；当前结果会按预期返回失败
python experiments/v043_team_mappo/scripts/check_promotion.py \
  --evaluation experiments/v043_team_mappo/metrics/v043_fixed_eval_0003.json
```

## 正式晋级顺序

1. 接入已经单独通过封存测试的传球、门将扑救和起身技能，使上层动作从纯速度扩展为“技能选择＋目标参数”。
2. 3v3先做教师模仿，再做 recurrent MAPPO 和对手联盟训练。
3. 用20场未见种子比赛检查胜率、摔倒、控球、真实传球完成率、失球和23自由度安全。
4. 3v3所有门槛通过后才进入5v5，再进入完整7v7十分钟比赛。

详细的 v035 差距审计见 `V035_AUDIT.md`，通信边界见 `COMMUNICATION_AUDIT.md`，所有晋级阈值见 `configs/v043_3v3_to_7v7.json`。
