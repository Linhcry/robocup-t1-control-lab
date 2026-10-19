# v046：精确 RCSS T1 全身定向传球基础任务

## 当前结论

`quality_passed=false`。v046 建立的是可在 MuJoCo-Warp 上并行训练的 23 DoF 全身定向传球环境，目的是替代 v040 以脚本教师为主体的训练路径；它不是已经学会传球的模型。接球仅保留下一阶段接口，调用 `stage="receive"` 会明确抛出 `NotImplementedError`，不会生成虚假成绩。

## 任务里真实存在什么

- 机器人直接复用 v045 已审计的 RCSS 安装包 `robots/T1/robot.xml` 构造：24 个机器人 body、36 个碰撞 geom、23 个关节和 23 个受限位置伺服器。
- 每个并行环境有一个质量 0.43 kg、半径 0.11 m 的自由球。球的初始位置、传球方向和传球距离分别独立随机化。
- `foot_ball_contact` 只匹配 `left_foot`、`right_foot` 与 `ball_geom` 的真实 MuJoCo 接触；成功奖励必须先出现该接触。
- `upper_body_ball_contact` 按刚体匹配躯干、腰、头、肩、全部手臂和手，因此也覆盖 RCSS XML 中没有单独名称的上臂碰撞体。靠身体撞球会立即失败，不能冒充脚传球。
- Actor 为 84 维，只包含可部署本体观测、球相对位置/速度、目标方向/距离和上一动作；Critic 为 101 维，额外读取精确球状态、接触、进度、机身线速度与高度。
- 动作为规范顺序的 23 维关节位置偏移。错误维度或 NaN/Inf 会被拒绝，目标在 GPU 上按 `locomotion/t1_spec.py` 的上下限逐环境裁剪。
- 倒地、硬关节越限、硬扭矩越限、上身触球和球出界均是独立终止条件；姿态、动作变化、扭矩使用也有连续惩罚。

## 训练逻辑

本版本只采用 [Daffan/humanoid-soccer](https://github.com/Daffan/humanoid-soccer) Apache-2.0 项目公开路线中的前两个训练思想：前期强调稳定接近并站到球后方，随后逐步提高朝目标方向的球速、前进距离、走廊精度和成功奖励。没有复制其 Isaac Gym 代码或任何权重，也没有使用受限 HTWK 权重。DAgger、N-P3O、视觉延迟/遮挡学生训练和接球属于后续独立版本，详见 `notes/provenance_and_scope.md`。

## 验证顺序

先在正式 WSL 中运行模型语义审计：

```bash
cd /home/linh/robot_soccer/training_runs/rcss_t1_training_20260706
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v046_fullbody_pass/scripts/audit_v046_model.py
```

再运行 1 个和 4 个 CPU 环境的 8 步动力学、动作保护和真实接触探针：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v046_fullbody_pass/scripts/smoke_v046.py
```

最后用 4 个 CPU 环境完成一次真实 PPO 更新。这个命令只证明训练链路可执行，JSON 始终保留 `quality_passed=false`：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v046_fullbody_pass/scripts/train_v046.py \
  --cpu --num-envs 4 --iterations 1 --seed 46040 \
  --run-name v046_cpu_4env_smoke_r1 --nan-guard --write-smoke-evidence
```

所有结果使用递增 `run_XX.json`，脚本拒绝覆盖旧证据。旧版 `configs/v046_competitive_acceptance.json` 作为历史合同保留；严格评测使用新增的 `configs/v046_competitive_acceptance_v2.json`。只有语义审计、CPU smoke、后续 GPU 扩展探针、正式训练、80 回合固定集、400 回合封存集和录像全部满足 v2 合同，本阶段才允许晋级。

## 严格 fixed / held-out 评测

`scripts/evaluate_v046.py` 不使用“球离开了就算踢到”的近似判断。每个 8 秒回合都逐策略步保存两只脚与球的真实 MuJoCo 接触、12 个上身刚体与球的非法接触、关节位置、实际 `qfrc_actuator` 扭矩、倒地和所有非有限值；脚和上身接触还会读取每个策略动作内完整的 4 个物理子步历史，短促踢球不会因只看最后一帧而漏报。传球结果同时保存目标距离、最远前向距离、方向误差、走廊横向误差和实际球高；严格成功必须先有真实脚触球，并且整个回合没有上身触球、倒地、关节越限、实际扭矩越限、非有限值或缺步。

固定集为种子 `460100..460179` 的 80 回合，封存集为 `461100..461499` 的 400 回合。每个种子确定一个方向/距离网格样本，证据绑定检查点、v2 合同、当前全部 Python 源码、语义审计、种子清单和情景清单哈希。封存集在同一检查点的固定集证明通过前会直接拒绝启动；旧证据不能覆盖，也不能替换新源码下的审计。

当前只运行了一个完整 8 秒 CPU 工程回放：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v046_fullbody_pass/scripts/evaluate_v046.py \
  --checkpoint <v046_model.pt> \
  --semantic-audit experiments/v046_fullbody_pass/metrics/v046_model_semantic_audit_run_07.json \
  --smoke --episodes 1 --seed-start 460001 --device cpu
```

正式固定集要去掉 `--smoke --episodes ...`；固定集通过后，封存集必须额外传入 `--split held_out_test --fixed-proof <固定集JSON>`。固定集或封存集的行为门槛即使通过，项目级 `quality_passed` 仍保持 `false`，直到独立录像和完整训练记录也齐全。

## 从 v045 形状感知迁移

v045 和 v046 共用相同的 `512-256-128` 网络及 23 维动作。`warmstart.py` 只做下列可逐元素审计的映射：Actor 把 v045 的前 75 个本体输入复制到 v046 前 75 列，把新增 9 个足球输入列设为零；Critic 把共享 75 列复制到原位置，把 v045 的 `base_lin_vel` 三列 `75:78` 移到 v046 的 `84:87`，其余新增列设为零。后续层与动作分布原样复制，Actor 新归一化器从恒等变换开始，Critic 归一化统计按同样索引映射；优化器动量、迭代数和课程计数全部清零。任何未知键、形状或参数顺序都会 fail-closed。

准备、独立审计和 CPU 一次更新的入口为：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v046_fullbody_pass/scripts/prepare_v046_warmstart.py \
  --source <v045_model.pt>

/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v046_fullbody_pass/scripts/audit_v046_warmstart.py \
  --source <v045_model.pt> --target <v046_warmstart.pt>

/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v046_fullbody_pass/scripts/train_v046.py \
  --cpu --num-envs 4 --iterations 1 --seed 46045 \
  --run-name v046_v045_warmstart_cpu_smoke_r1 --nan-guard \
  --write-smoke-evidence \
  --warmstart-checkpoint <v046_warmstart.pt> \
  --warmstart-audit <passing_warmstart_audit.json>
```

## 正式训练入口（本轮未启动）

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v046_fullbody_pass/scripts/train_v046.py \
  --num-envs 4096 --iterations 3000 --seed 46000 \
  --run-name v046_directional_pass_full_r1 --nan-guard
```

本轮按要求不启动 GPU 长训。正式训练前仍需单独运行 1、16、4096 环境的 GPU 扩展探针，并确认不会影响正在运行的起身训练。

## 本轮实际证据

- `metrics/v046_model_semantic_audit_run_07.json`：26 项结构检查全部通过，精确 RCSS 机器人为 24 body、36 geom、23 关节/动作；完整上身刚体、4 个物理子步接触历史、80/400 回合种子集合、迁移索引以及训练/评测/比赛控制共用的 `TORQUE_LIMIT` 都通过审计；`model_semantics_passed=true`，但 `quality_passed=false`。
- `metrics/v046_cpu_dynamics_smoke_run_02.json`：1/4 环境各运行 8 个策略步；Actor 84 维、Critic 101 维；脚接触探针真实锁存，上身接触探针真实终止；所有检查通过，但 `quality_passed=false`。
- `metrics/v046_cpu_ppo_smoke_run_02.json`：4 个 CPU 环境完成 96 个采样步和一次 PPO 更新，生成新的 `model_0.pt`；这不是技能检查点，训练与验证仍为假。
- `metrics/v046_t1_contract_audit_run_01.json`：新增控制路径后重新运行中央 23 DoF 审计，结果为 `PASS: T1_23_DOF_CONTRACT`。
- `metrics/v046_v045_warmstart_audit_run_05.json`：在当前源码清单下，逐元素映射、严格检查点加载和一次 CPU PPO 更新全部通过；Actor 新 9 列及 Critic 新列为零，后续层/分布完全复制，优化器/迭代/课程已重置；`quality_passed=false`。
- `metrics/v046_cpu_ppo_smoke_run_05.json`：转换后的 84/101 维检查点在当前 4 子步接触锁存和保守比赛扭矩上限下被 RSL 严格加载，4 个 CPU 环境完成 96 个采样步和一次真实更新，输出张量形状与有限性再次通过；这不是传球成绩。
- `metrics/v046_model_0_engineering_smoke_run_04.json`：在当前源码与 `run_07` 语义审计绑定下，严格评测器完成 1 个 8 秒、400 策略步回放并保存全部接触/关节/实际扭矩证据；该未训练模型没有脚触球，并出现上身触球、倒地、关节和保守扭矩上限违规，传球成功率为 0，因此 `quality_passed=false`。

最初的 `run_01` 只按有名称的上身 geom 检查，独立复核后发现会漏掉 RCSS XML 中未命名的肩和上臂碰撞体，因此已经由完整刚体过滤器和 `run_02` 替代；`metrics/v046_contact_filter_supersession_20260820.json` 明确禁止旧证据晋级。

目前没有固定集、封存集、有效传球模型或接球模型，不能用于比赛或晋级。
