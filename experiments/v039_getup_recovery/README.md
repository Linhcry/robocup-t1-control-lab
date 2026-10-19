# v039 T1 倒地起身与恢复

本目录是一条独立的底层能力线，不修改 v038，也不把“训练程序跑完”写成“起身已经达标”。控制对象是完整 23 自由度 T1；每个目标角都经过 `clamp_t1_targets`，动作维度、有限值、执行器顺序和力矩上限都在环境加载时检查。

## 当前可证实的官方边界

- [Booster booster_train](https://github.com/BoosterRobotics/booster_train/tree/651b7a53f2ffaf2d5629d0604d065cc385e29c6b) 当前公开的 BeyondMimic 机器人任务只有 K1，没有 T1 起身任务。
- [Booster booster_assets](https://github.com/BoosterRobotics/booster_assets/tree/508cbee6ca9ae6fbc8c0b38dd58785a6f3fc61a2) 有 T1 23DoF 模型，但公开动作 CSV 只有两段 K1 动作，没有 T1 起身轨迹。
- [Booster booster_deploy](https://github.com/BoosterRobotics/booster_deploy/tree/563e7c37e7468b3b96ad9f84f9e61becea3b7fb8) 公开了 `t1_walk.pt`，没有 T1 起身策略。
- [Booster booster_gym](https://github.com/BoosterRobotics/booster_gym/tree/da396a06d6eed99e2de72d7749c48ee8748950f9) 的公开 T1 环境只训练行走，并在机身高度低于倒地阈值时终止回合，没有倒地后继续起身的任务。
- [Booster RoboCup demo](https://github.com/BoosterRobotics/robocup_demo/blob/fa4f31ccb4428a9e60e90b4f54eb80499caeb0ca/src/brain/src/robot_client.cpp) 的比赛起身调用固件 `LocoApiId::kGetUp`；公开仓库没有给出这段固件动作的关节轨迹或权重。

因此，真机比赛优先调用官方固件起身；MuJoCo/RCSS 的自研策略必须独立训练和封存测试，二者不能冒充同一个模型。

## 课程和达标门槛

1. 第0阶段：8°–25°临倒恢复。
2. 第1阶段：左右侧卧起身。
3. 第2阶段：俯卧、仰卧起身。
4. 第3阶段：五类姿态混合，并随机地面摩擦、初始关节和速度。

每阶段必须先通过固定验证，再允许进入下一阶段。最终封存集严格按总验收合同执行：仰卧、俯卧、左侧卧、右侧卧各 100 个样本；单类成功率至少 95%，总体至少 97%；中位起身时间不超过 5 秒、95 分位不超过 8 秒；起身后连续站立 3 秒并按随机方向行走 5 秒的完整链路至少 95%。危险接触率和关节越限率必须为 0，并且每一步都发送全部 23 路安全目标。未达到这些数字时，状态必须写为失败或候选。

## 在 WSL 中运行

```bash
cd /home/linh/robot_soccer/training_runs/rcss_t1_training_20260706
source /home/linh/robot_soccer/envs/rl-mujoco/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/audit_t1_23dof.py
python experiments/v039_getup_recovery/scripts/audit_v039.py
python experiments/v039_getup_recovery/scripts/audit_joint_limits_and_semantics.py
python experiments/v039_getup_recovery/scripts/smoke_v039.py
python experiments/v039_getup_recovery/scripts/audit_handoff.py
python experiments/v039_getup_recovery/scripts/evaluate_stage.py --stage 0 --zero-action
python experiments/v039_getup_recovery/scripts/train_stage.py --stage 0 --timesteps 4096
python experiments/v039_getup_recovery/scripts/evaluate_stage.py --stage 0 --checkpoint experiments/v039_getup_recovery/checkpoints/v039_stage_00_iter_0001.zip
```

固定验证默认使用从 `39100` 开始的种子，封存测试默认使用独立的 `1039100` 种子空间；`--held-out` 不再只是更换文件名，并会拒绝与固定验证范围重叠的种子。第1阶段以后必须同时提供 `--resume` 和上一阶段由本评估脚本生成且 `quality_passed=true` 的 `--previous-stage-proof` JSON，脚本会核对阶段号、评估类型和检查点路径，不能再仅凭“传了一个模型文件”绕过课程门槛。例如：

```bash
python experiments/v039_getup_recovery/scripts/train_stage.py \
  --stage 1 \
  --resume experiments/v039_getup_recovery/checkpoints/v039_stage_00_iter_XXXX.zip \
  --previous-stage-proof experiments/v039_getup_recovery/metrics/v039_stage_00_iter_XXXX_fixed_eval.json
```

`--held-out` 只在固定验证通过后使用，避免反复观察封存集造成过拟合。

## 2026-08-20 实际结果

- `v039_stage_00_iter_0001.zip`：4096 PPO 步，固定 20 种子 0/20。
- `v039_stage_00_iter_0002.zip`：在第一个模型上继续约 10 万步，固定 20 种子仍为 0/20。分析发现旧奖励允许机器人倒地后累积正回报，因此这个检查点只作为失败证据。
- 随后把奖励改为“躯干高度与直立度的乘积”，倒地状态变为负回报，并降低初始策略方差；从头训练 `v039_stage_00_iter_0003.zip` 约 10 万步，固定 20 种子仍为 0/20，不能晋级。
- 完整 23DoF 模型确实具有 23 个执行器，上肢和腰部不是“接口没接上”；但从理想直立状态开始，静态准备姿态 PD 在 1.04 秒失去严格稳定，公开的官方 `T1.pt` 腿部策略接到全身模型后在站立阶段 0.56 秒失去严格稳定，二者都未进入 5 秒行走阶段。具体数值见 `metrics/v039_stand_walk_handoff_audit.json`。
- 为避免浪费封存集，只做了每姿态 5 个固定诊断种子：当前 `iter_0003` 在左侧卧、右侧卧、俯卧、仰卧均为 0/5，见第二份 `v039_stage_00_iter_0003_fixed_eval_run_02.json`；这不是最终 100×4 封存测试。
- 独立关节审计确认所有 23 路发送目标都经过两次限位裁剪，极端正负动作的发送目标越限为 0；但是物理关节状态会在 MuJoCo 的软限位下真实越界。固定种子 `39910` 的右侧卧在 reset 完成时 `Right_Hip_Roll` 已越界约 `0.0482 rad`，俯卧的 `Head_pitch` 已越界约 `0.0280 rad`，随后膝关节也会出现负角。旧代码在 0.30 秒无控制自由落体 settle 后把越限标志清零，属于漏报；现已保留 reset 越限并把“发送目标越限”和“物理状态越限”分开记录。证据见 `metrics/v039_joint_limit_and_model_semantics_audit_run_02.json`。
- `T1_serial` 与 RCSS 完整 T1 的 23 关节顺序及 XML 限位完全一致，但动力学和控制接口不等价：前者为 23 个直扭矩电机、2 ms 步长、总质量约 35.18 kg；后者每关节使用 torque/position/velocity 三执行器、5 ms 步长，机器人约 31.61 kg，且摩擦、躯干和足底碰撞体不同。官方 `T1.pt` 又是在固定上身的 12 腿关节 `T1_locomotion` 上训练，因此当前 handoff 失败属于模型/执行器适配失败，不能通过改统计口径变成通过。

当前状态是 `quality_passed=false`。固件 `kGetUp` 仅能作为真机比赛时的官方能力调用，不能写成我已经训练出的公开起身策略。

## 当前机器为什么不直接安装 Isaac Lab

当前 WSL 只有约 8 GB 内存配额，RTX 5070 Laptop 只有约 8 GB 显存。NVIDIA [Isaac Sim 5.0 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html) 给出的最低规格为 32 GB 内存和 16 GB 显存，Isaac Lab 训练还需要更多资源。强行安装十几 GB 软件并不能让不满足硬件门槛的训练变成可靠结果，所以 v039 先使用已实际可运行的 MuJoCo；若要进行数千并行环境的全姿态训练，需要换到至少 32 GB RAM、16 GB VRAM 的 Linux/工作站，再移植到 Isaac Lab。
