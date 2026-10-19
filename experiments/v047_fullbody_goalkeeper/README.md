# v047 精确 RCSS T1 全身门将

`v047` 是一个独立的新实验，不修改也不覆盖 `v042` 或 `v045`。它把 `v045` 已审计的精确 RCSSServerMJ T1 机器人放入 MuJoCo-Warp 并行环境，使用全部 23 个关节直接学习横移、屈膝、伸腿、转体和保持平衡等全身封堵动作。足球的半径、质量、摩擦、接触维度和接触求解参数来自已安装的 RCSSServerMJ `soccer/world.xml`。

## 当前真实状态

| 证据 | 结果 |
|---|---|
| `metrics/v047_canonical_23dof_audit_run_01.json` | 项目统一 23 DoF 顺序、马达映射、执行器和限位审计通过 |
| `metrics/v047_target_contract_audit_run_12.json` | 精确机器人、足球、动作顺序、接触传感器、四阶段课程、120/360 种子合同、RCSS 空气参数和 v045→v047 映射语义全部通过结构审计 |
| `metrics/v047_cpu_smoke_8_steps_run_06.json` | 真实 CPU 物理运行 8 步；Actor `1×87`、Critic `1×107`、Action `23`，所有张量有限；并真实复位 27 个评估格子、验证同种子同格子可精确复现 |
| `metrics/v047_warmstart_cpu_ppo_smoke_run_06.json` | 严格核对逐列映射、哈希、优化器/迭代/课程清零，目标网络严格加载，并在 CPU 完成一次使 Actor 与 Critic 参数真实变化且仍有限的 PPO 更新 |

这些证据只证明“训练任务、迁移和一次小更新可以正确构造和运行”，没有证明门将能力。工程冒烟使用的是尚未完成正式评测的 v045 `model_50.pt`，它和由它生成的 v047 检查点都被明确标为不可晋级。正式 v045 训练总结与候选证明尚未生成，因此正式 v047 入口当前会 fail-closed。GPU 并行探针、PPO 长训练、120 球固定验证、360 球封存测试和视频网格也尚未运行，所以当前以及所有现有 JSON 都保持 `quality_passed=false`。

## v045 起身策略的严格迁移

v045 Actor 的 75 维输入依次是 `3` 维基座角速度、`3` 维重力方向、`23` 维关节位置、`23` 维关节速度和 `23` 维上一动作。v047 把这 75 列原样放在最前面，并把新增的 `6` 维球状态、`4` 维交点预测和 `2` 维门线位置共 12 列初始化为精确零。因此 warm-start 刚加载时，新球信息不会无意改变已学到的全身动作，但 PPO 可以从第一轮开始为这些列学习非零权重。

v045 Critic 的前 75 维同样复制；它原来位于 `75:78` 的三维特权基座线速度被移到 v047 的 `87:90`，对应的归一化均值、方差和标准差也一同移动。其余球场景、安全和接触力列全部从零权重开始。Actor/Critic 的后续隐藏层、输出层和高斯动作分布逐张量复制；优化器动量清空，学习率恢复到 v047 的 `3e-4`，迭代号和 `common_step_counter` 都归零，因此不会继承 v045 的晚期课程阶段。

为了让“复制前 75 列”不仅形状相同而且含义相同，v047 现在保持 v045 的关节速度尺度、Actor 无经验归一化、Critic 有经验归一化、Actor 使用带编码器偏置的关节位置而 Critic 使用无偏置关节位置，并使用相同的 `0.60` 相对关节目标尺度。`warmstart.py` 同时拒绝未知网络键、变化后的层形状、非有限权重、错误优化器参数顺序和任何不匹配的证明哈希。

正式迁移必须同时绑定完整的 `v045_mjlab_training_summary_v1` 和正式 `v045_target_candidate_eval_v1`：训练必须是 4096 环境、3000 轮完整运行，候选筛选必须是种子 `45100/45101/45102`、每种子 256 环境、299 步，且选中的路径和 SHA-256 必须逐项一致。工程证明不能代替这两个正式文件。

当前 `--resume` 只允许配合 `--engineering-smoke` 使用。正式续训还没有不可变的 v047 训练血缘证明格式，因此先 fail-closed，不能通过把工程检查点放进日志目录来绕开 v045 正式来源校验。

## 学习问题

Actor 只能看到部署时可获得的信息：基座角速度、重力方向、23 维关节位置、23 维关节速度、上一帧 23 维动作、机器人坐标系内的球位置与速度、球路和门将平面的交点预测，以及里程计给出的门线坐标系纵深/横向位置。它看不到奖励中的碰撞真值、采样目标或安全标签。Critic 在训练期额外读取基座线速度、射门采样真值、接触力和安全状态，因此这是“局部可部署 Actor + 特权 Critic”的非对称 PPO。

球路预测只采用防守线交点这一可解释几何思想：当球向门将运动时，按当前速度外推它穿过门将平面的时间、横向位置和高度。该思路与 HTWK K1 `Ball_Intercept` 所代表的门将预判方向一致，但本实验没有复制 HTWK 权重、受限模型或实现；23 DoF 网络、RCSS 碰撞体、奖励和验收均在本项目内独立建立。

严格接触证据把事件分为三类：

- 合法封堵：脚、 小腿、躯干或大腿必须真实碰球；只有球连续 0.30 秒不再危险地飞向门内，才记为封堵。
- 危险冲击：头部与手/前臂分别记录球和地面的峰值接触力；超过 350 N 或 300 N 即失败。
- 挡后稳定：封堵后必须连续站立 3 秒，同时没有倒地、关节越限、非有限状态或危险冲击，才记一次完整成功。

此外每一步都记录失球、倒地、硬限位、有效扭矩上限饱和和全身动作维度。旧的“碰到球就算成功”或把不连续安全帧相加的口径不能通过本合同。

## 四阶段课程

1. 门心附近、3.5–4.5 m/s 的直线地滚球。
2. 更宽的横向目标、3.5–6.0 m/s 和随机射门角度。
3. 全门宽、4.0–8.0 m/s、地滚球和最高 0.90 m 的空中球。
4. 全门宽、3.0–9.0 m/s、最高 1.12 m，并加入起点、角度、自旋、门将位置/朝向/初速度和关节姿态扰动。

完整数值合同在 `configs/v047_competitive_acceptance.json`。固定验证使用 120 个种子；只有同一检查点先通过固定验证，才允许运行与之完全不重叠的 360 个封存种子。封存集还要求横向 × 速度 × 高度 27 个格子每格至少 8 球，并要求独立视频网格证明。局部课程成绩不能自动升级为完整门将成绩。

## 文件入口

- `rcss_goalkeeper_task.py`：精确机器人/足球、87/107 维观测、奖励、接触传感器、课程和 PPO 配置。
- `mdp.py`：23 维有限检查与批量限位、射门复位、交点预测、严格事件状态机、奖励和指标。
- `scripts/audit_v047_task.py`：结构和验收合同审计，输出永不覆盖的 JSON。
- `scripts/smoke_v047.py`：真实 CPU 8 步冒烟，输出永不覆盖的 JSON。
- `scripts/train_v047.py`：4096 环境 GPU PPO 入口；本次没有执行长训练。
- `warmstart.py`：v045 `75/78` 维网络到 v047 `87/107` 维网络的逐列迁移、正式证明校验与内容寻址缓存。
- `scripts/create_engineering_source_proof.py`：只为 CPU 管线测试生成明确不可晋级的工程来源证明。
- `scripts/smoke_v047_warmstart.py`：逐张量审计迁移并真实执行一次小型 CPU PPO 更新。
- `scripts/evaluate_v047.py`：固定/封存确定性评估；封存模式强制检查同一模型的固定验证证明和视频证明。

## 运行顺序

在正式 WSL 项目目录执行：

```bash
cd /home/linh/robot_soccer/training_runs/rcss_t1_training_20260706
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

/home/linh/robot_soccer/envs/rl-mujoco/bin/python scripts/audit_t1_23dof.py
/home/linh/robot_soccer/envs/mjlab-playground/bin/python experiments/v047_fullbody_goalkeeper/scripts/audit_v047_task.py
/home/linh/robot_soccer/envs/mjlab-playground/bin/python experiments/v047_fullbody_goalkeeper/scripts/smoke_v047.py
```

GPU 资源空闲后，先做小规模并行探针，检查显存、吞吐和 NaN；它也不能晋级：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v047_fullbody_goalkeeper/scripts/train_v047.py \
  --num-envs 256 --iterations 1 --seed 47002 \
  --run-name v047_gpu_probe_256_r1 --nan-guard \
  --warmstart-v045-checkpoint /absolute/path/to/selected_v045_model_N.pt \
  --v045-training-summary /absolute/path/to/v045_training_summary.json \
  --v045-candidate-proof /absolute/path/to/v045_target_candidate_proof.json
```

只有探针通过后才运行正式 4096 环境训练：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v047_fullbody_goalkeeper/scripts/train_v047.py \
  --num-envs 4096 --iterations 5000 --seed 47000 \
  --run-name v047_fullbody_goalkeeper_full_r1 --nan-guard \
  --warmstart-v045-checkpoint /absolute/path/to/selected_v045_model_N.pt \
  --v045-training-summary /absolute/path/to/v045_training_summary.json \
  --v045-candidate-proof /absolute/path/to/v045_target_candidate_proof.json
```

训练完成后必须先固定验证，再把其 JSON 作为封存评估证明：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v047_fullbody_goalkeeper/scripts/evaluate_v047.py \
  --checkpoint /absolute/path/to/model_XXXX.pt \
  --target-audit /absolute/path/to/v047_target_contract_audit_run_07.json

/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v047_fullbody_goalkeeper/scripts/evaluate_v047.py \
  --checkpoint /absolute/path/to/model_XXXX.pt --held-out \
  --target-audit /absolute/path/to/v047_target_contract_audit_run_07.json \
  --fixed-proof /absolute/path/to/v047_policy_fixed_validation_run_XX.json \
  --video-proof /absolute/path/to/v047_video_grid_proof_run_XX.json
```

封存评估不会只相信一个手写的 `quality_passed`。固定证明必须同时匹配检查点、验收合同、当前 `mdp.py`/任务源码和审计文件的 SHA-256，并包含完整且按顺序的 120 条种子记录。视频证明采用 `v047_video_grid_proof_v1`：同一检查点必须有 27 个不重复格子的本地视频，每条记录给出 `cell: [横向格, 速度格, 高度格]`、绝对 `path` 和文件 `sha256`；评估器会逐个重新计算哈希。

## 尚未训练的边界

`v047` 只负责一个门将面对一脚来球的低层全身封堵。它没有训练控球后传球、出击时与后卫冲突、连续第二点、点球大战策略、3v3/7v7 角色决策或摔倒后的独立起身切换；这些能力必须分别通过传球、团队决策和起身检查点后再集成。当前只有用于验证加载和更新的工程检查点，没有任何可晋级的 v047 神经网络检查点，不能用于比赛展示或宣称具备竞争力。

## 参考边界

- [MJLab](https://github.com/mujocolab/mjlab)：本任务使用的管理式环境和 MuJoCo-Warp GPU 并行基础。
- [MJLab Playground](https://github.com/mujocolab/mjlab_playground)：T1 23 DoF 训练配置的官方上游来源。
- [HTWK Gym](https://github.com/NaoHTWK/htwk-gym)：只参考并行训练、部署观测与球类任务的公开设计方向；其 README 明确给预训练模型设置竞赛使用限制，本项目没有导入这些权重。
