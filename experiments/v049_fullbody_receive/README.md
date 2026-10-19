# v049：精确 RCSS T1 全身接球与停球基础任务

## 当前结论

`quality_passed=false`。v049 已建立可在 MuJoCo-Warp 上 GPU 并行的 23 DoF 全身接球/停球训练与严格评测链路，并完成 v045→v049 的 shape-aware 工程 warm-start、CPU 真实接触和一次 PPO 更新检查；本轮没有启动 GPU 长训练，也没有产生具备接球能力的模型。当前 v045 只有工程 probe，没有正式训练总结与正式候选证明，因此正式 warm-start 和正式训练入口保持 fail-closed。

## 这次训练的“接住球”具体指什么

每回合会在机器人前方生成一个 RCSS 真实物理球，并给球施加朝机器人方向的线速度、可能的竖直速度和旋转。球必须先真实碰到 `Shank_Left`、`left_foot_link`、`Shank_Right` 或 `right_foot_link`，其余 20 个机器人刚体碰球全部算非法。成功还必须同时满足：第一次合法触球不是等球自然停下后轻碰；球速相对触球前至少降低 50%；球的绝对速度不大于 0.65 m/s；球相对机器人的平面速度不大于 0.45 m/s；球在机器人 0.55 m 可控半径内；这些条件连续保持至少 1.0 秒。瞬时碰球、距离接近或回合超时都不能算成功。

倒地、关节越限、超过 RCSS 逐关节有效扭矩上限、头/手/躯干危险撞地、任何非法身体触球、数值异常和漏球都会锁存为失败。脚/小腿和其余身体按 RCSS body 划分，所以模型中没有名字的肩、上臂、髋和踝部碰撞体也不会漏检。

## 深度学习输入和强化学习课程

Actor 为 81 维，只包含机器人角速度、重力方向、23 个关节位置、23 个关节速度、上一帧 23 维动作，以及带噪声和 0–2 帧延迟的球相对位置/速度。这些量可由实机本体传感器与视觉/状态估计提供。Actor 看不到接触锁存、真实世界坐标、来球课程标签或成功状态。Critic 为 109 维，在 Actor 信息之外可以使用精确球状态、来球速度/角度/落点和安全状态，以非对称 Actor-Critic 提高训练效率。

课程分四级逐步扩大三个轴：来球速度、水平来球角和物理落点。第 0 级是 0.9–1.4 m/s 的中央低速地滚球；第 3 级扩大到 1.0–4.5 m/s、±45°、左右 0.52 m 落点，并加入低弹跳和旋转。正式训练仍需从第 0 级开始，不能把尚未通过的高难度结果写成完成。

## v045→v049 shape-aware warm-start

warm-start 脚本不是按“看起来维度差不多”硬拷贝，而是同时实例化真实 v045、v049 环境，从 ObservationManager 读取每个观测项的实际名称、函数、形状和切片，任一布局不符合合同便终止。当前实测布局与变换为：

- v045 Actor 为 75 维，v049 Actor 为 81 维。第一层权重的源 `[0:75]` 精确复制到目标 `[0:75]`，新增球相对位置 `[75:78]` 和球相对速度 `[78:81]` 六列全部置零。
- v045 Critic 为 78 维，v049 Critic 为 109 维。共同的源 `[0:75]` 精确复制到目标 `[0:75]`；v045 的 `base_lin_vel` 源 `[75:78]` 精确映射到 v049 的 `base_linear_velocity_privileged` 目标 `[81:84]`；目标 `[75:81]` 与 `[84:109]` 等其余新增列全部置零。
- v049 Critic 的共同 `joint_pos` 已固定为和 v045 Critic 一样的无偏置语义。其余形状兼容的隐藏层、输出层、bias 和 Actor 动作分布参数逐张量复制。
- v045 Actor 检查点没有 Actor normalizer，所以 v049 Actor normalizer 以 `mean=0, var=1, std=1, count=0` 中性初始化；Critic 的共同 75 维统计量和三维 base linear velocity 统计量按相同切片映射，所有新增维使用中性统计量。
- 目标优化器状态清空，学习率重置为 `0.0003`，迭代号和课程 `common_step_counter` 都重置为 0。一次 CPU PPO 更新后，计数器前进到 32、优化器才产生 17 个新状态项，证明不是继续沿用 v045 的优化器或课程进度。

每个 warm-start 产物都绑定源 checkpoint、v045 训练总结、v045 候选证明、目标模板和关键源文件 SHA-256。正式模式只接受互相反向绑定的 `v045_training_summary_v1` 与 `v045_v049_candidate_selection_v1`，并要求正式训练已完整结束；当前工程 summary/probe 即使文件存在也不能通过正式模式。`--engineering-smoke` 只能配合 CPU、最多 4 个环境、恰好 1 次迭代和 smoke 证据写入，不能用来启动长训练。

## 本轮验证证据

- `artifacts/v049_v045_warmstart_engineering_run_02.pt` 与同名 `.provenance.json`：当前 shape-aware 工程 warm-start 及完整来源、布局、映射和重置证明；`formal_training_allowed=false`、`quality_passed=false`。
- `metrics/v049_warmstart_semantic_audit_run_01.json`：独立重新读取源/目标张量，验证 Actor 75→81、Critic 78→109、base linear velocity 75:78→81:84、其余新增列为零，并实际验证正式准备和正式训练都会拒绝工程证据。
- `metrics/v049_warmstart_cpu_ppo_smoke_run_02.json`：4 个 CPU 环境严格加载当前 warm-start，完成 128 个采样步和一次 PPO 更新；它只证明加载与更新链路可运行，不是接球技能证据。
- `metrics/v049_model_semantic_audit_run_04.json`：精确 RCSS 机器人、23 维动作、球参数、身体接触全集、Actor/Critic 隔离、Critic 共同切片语义、触球前减速指标归零、1 秒状态机、课程和防伪合同全部通过；`quality_passed=false`。
- `metrics/v049_cpu_dynamics_smoke_run_04.json`：对当前最终源文件，1/4 个 CPU 环境完成真实动力学；脚和小腿碰球被合法传感器捕获，躯干碰球立即失败，单次碰球不能冒充 1 秒停球；`quality_passed=false`。
- `metrics/v049_cpu_ppo_smoke_run_01.json`：4 个 CPU 环境完成 128 个采样步和一次 PPO 更新，并生成 `model_0.pt`。它只证明训练链路可运行，不是技能模型。
- `metrics/v049_t1_contract_audit_run_02.json`：中央 23 DoF、执行器通道、有限控制值和关节限位复核通过。
- `metrics/v049_evidence_sync_correction_20260820.json`：如实记录第一次临时 smoke 证据未归档的同步问题和修正，禁止把临时失败结果隐去后宣称能力达标。
- `metrics/v049_warmstart_current_evidence_supersession_20260820.json`：按 SHA-256 指定上述 warm-start、语义、真实接触和 23 DoF 文件为当前工程证据；被替代的 run 只保留历史，不得用于晋级。

## 运行方式

在正式 WSL 中先运行语义审计与 CPU 真实接触检查：

```bash
cd /home/linh/robot_soccer/training_runs/rcss_t1_training_20260706
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v049_fullbody_receive/scripts/audit_v049_model.py
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v049_fullbody_receive/scripts/smoke_v049.py
```

CPU 一次 PPO 更新只用于工程检查：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v049_fullbody_receive/scripts/train_v049.py \
  --cpu --num-envs 4 --iterations 1 --seed 49040 \
  --run-name v049_cpu_4env_smoke_r1 --nan-guard --write-smoke-evidence
```

当前工程 warm-start 的独立复核命令如下：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v049_fullbody_receive/scripts/audit_v049_warmstart.py \
  --artifact experiments/v049_fullbody_receive/artifacts/v049_v045_warmstart_engineering_run_02.pt \
  --provenance experiments/v049_fullbody_receive/artifacts/v049_v045_warmstart_engineering_run_02.provenance.json \
  --ppo-proof experiments/v049_fullbody_receive/metrics/v049_warmstart_cpu_ppo_smoke_run_02.json
```

只有将来正式 v045 训练完整结束、生成正式 summary 和候选证明之后，才允许生成正式 warm-start。省略 `--engineering-smoke` 会启用严格正式合同：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v049_fullbody_receive/scripts/prepare_v049_warmstart.py \
  --source-checkpoint /absolute/path/to/formal_v045_checkpoint.pt \
  --target-template /absolute/path/to/v049_structural_template.pt \
  --v045-training-summary /absolute/path/to/v045_training_summary.json \
  --v045-candidate-proof /absolute/path/to/v045_v049_candidate_selection.json
```

正式 GPU 训练入口已经准备好，但本轮按任务范围没有运行；它还必须同时提供上一步正式产物和 proof，工程产物会被拒绝：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v049_fullbody_receive/scripts/train_v049.py \
  --warm-start /absolute/path/to/formal_v049_warmstart.pt \
  --warm-start-proof /absolute/path/to/formal_v049_warmstart.provenance.json \
  --num-envs 4096 --iterations 4000 --seed 49000 \
  --run-name v049_fullbody_receive_full_r1 --nan-guard
```

正式训练完成后必须先固定集，再封存集。封存集要求同一检查点、同一合同、同一源文件和同一语义审计的固定集通过证明，同时还要 27 个速度×角度×落点单元的哈希视频清单：

```bash
/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v049_fullbody_receive/scripts/evaluate_v049.py \
  --checkpoint /absolute/path/model_N.pt \
  --semantic-audit experiments/v049_fullbody_receive/metrics/v049_model_semantic_audit_run_XX.json

/home/linh/robot_soccer/envs/mjlab-playground/bin/python \
  experiments/v049_fullbody_receive/scripts/evaluate_v049.py \
  --checkpoint /absolute/path/model_N.pt --held-out \
  --semantic-audit experiments/v049_fullbody_receive/metrics/v049_model_semantic_audit_run_XX.json \
  --fixed-proof experiments/v049_fullbody_receive/metrics/v049_policy_fixed_validation_run_XX.json \
  --video-proof /absolute/path/v049_video_grid_proof.json
```

固定集为 162 回合，封存集为 540 回合，分别覆盖 27 个速度×角度×落点单元。只有封存集、所有安全门、最差单元和视频证据共同通过，才允许把 `quality_passed` 变为真。
