# v043 结果清单

## 结论

团队训练基础设施已经能够安全运行3v3并参数化到7v7，但当前神经网络没有比赛竞争力，不能晋级。`local_v2` 固定种子短评估为2局0胜、2平、0负，比分均为0:0，平均球推进0米；严格晋级门槛的最终结果是 `quality_passed=false`。

## 失败与修复证据

1. 最早的 `v043_mappo_0001.pt`、`0002.pt` 把全局真值角色作为 Actor 输入，违反分散执行边界。`metrics/v043_global_role_leak_audit_0001.json` 将其判为不可部署，旧格式由 v2 加载器拒绝。
2. 第一次去除全局角色后，`metrics/v043_3v3_smoke_0002.json` 暴露两个外场球员同时选择压迫角色，角色契约失败。该结果未删除。
3. `local_v2` 改为由公开场地区域、球衣号和本机球可见性确定压迫权；全局接触、全局角色和全局状态仅供教师、奖励、指标与集中式 Critic。
4. `metrics/v043_decentralized_execution_audit_0002.json` 通过行隔离、Critic 隔离、触球不广播、视野外队友不影响角色和无 IPC 五项检查。

## 当前通过的工程门槛

- `metrics/v043_3v3_smoke_0003.json`：6/6站立，138路目标全部发送，有限值通过。
- `metrics/v043_7v7_smoke_0002.json`：14/14站立，322路目标全部发送，有限值通过。
- `metrics/v043_mappo_local_v2_training_0001.json`：启发式教师对手，8环境步训练链通过。
- `metrics/v043_mappo_local_v2_training_0002.json`：历史神经网络快照对手，8环境步训练链通过；证明对手池确实加载旧快照。

这些门槛只证明代码和控制安全，不证明团队能力。

## 当前失败的比赛门槛

`metrics/v043_fixed_eval_0003.json`：2局均为0:0，胜率0，平均球推进0米，真实传球完成率没有可验证样本。`metrics/v043_promotion_gate_0001.json` 拦截了以下六项：20场封存比赛数、胜率、真实传球完成率、十分钟失球指标、至少5类对手、已验证团队技能接口。摔倒、23自由度发送、短局控球比例、通信审计和 v2 观测格式通过，但不足以晋级。

## 完整复现命令

```bash
cd /home/linh/robot_soccer/training_runs/rcss_t1_training_20260706
source ~/robot_soccer/envs/rl-mujoco/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/audit_t1_23dof.py
python experiments/v043_team_mappo/scripts/audit_decentralized_execution.py
python experiments/v043_team_mappo/scripts/smoke_v043.py --players-per-team 3 --steps 8 --prepare-seconds 0.1
python experiments/v043_team_mappo/scripts/smoke_v043.py --players-per-team 7 --steps 8 --prepare-seconds 0.1

python -m experiments.v043_team_mappo.train_mappo \
  --players-per-team 3 --updates 1 --rollout-steps 8 --epochs 1 \
  --snapshot-interval 1 --episode-high-level-steps 16 --prepare-seconds 0.1

python -m experiments.v043_team_mappo.train_mappo \
  --players-per-team 3 --updates 1 --rollout-steps 8 --epochs 1 \
  --snapshot-interval 2 --episode-high-level-steps 16 --prepare-seconds 0.1 \
  --teacher-probability 0 \
  --resume experiments/v043_team_mappo/checkpoints/v043_mappo_local_v2_0001.pt

python -m experiments.v043_team_mappo.evaluate \
  --checkpoint experiments/v043_team_mappo/checkpoints/v043_mappo_local_v2_0002.pt \
  --players-per-team 3 --episodes 2 --episode-high-level-steps 16 --prepare-seconds 0.1

python experiments/v043_team_mappo/scripts/check_promotion.py \
  --evaluation experiments/v043_team_mappo/metrics/v043_fixed_eval_0003.json
```

最后一条命令当前应返回退出码1，这是严格门槛正确阻止未达标模型晋级，而不是程序故障。
