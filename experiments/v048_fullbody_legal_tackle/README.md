# v048 精确 RCSS 双 T1 全身合法抢断基础任务

v048 是从失败的 v041 独立重建的新实验，不修改也不继承 v041 的三维高层残差环境。每个并行世界都实际包含两台由 RCSSServerMJ 安装包 `robot.xml` 编译的 T1、每台 23 个受控关节，以及与正式足球环境相同的 0.11 米、0.41 千克足球。防守者策略直接输出 23 维全身关节目标；攻击者是受限外力驱动的物理教师，方向在 `[-π, π]`、速度在 `[0.15, 0.65] m/s` 随机化。它可用于训练环境建设，但还不是已经学会稳定行走带球的对手。

## 修复的旧漏洞

- 两台机器人位于同一个 MuJoCo 世界，机器人—机器人碰撞未关闭。独立重叠碰撞探针测得真实冲击 `454.025 N`，不是脚本布尔值。
- 防守者和攻击者的足/小腿—球接触分别由 MuJoCo contact sensor 采集；只有攻击者先合法触球、之后防守者单独合法触球，才记为换权。双方同时触球不算转移。
- 清球按随机攻击方向投影，成功线固定为远离己方球门 `0.25 m`，不再遗漏配置门槛。
- 抢断后必须在最近 0.15 秒内有防守者合法触球、球距不超过 0.55 米，并连续控制至少 2 秒。
- 双向的头、躯干、前臂或手与对手碰撞直接记犯规；跨机器人总接触力另行记录。
- 任一机器人倒地、关节越限、有效关节力矩超限、NaN/Inf 都是失败，并立即停止该回合。

## Actor 与 Critic

Actor 只有可部署的 93 维局部信息：防守者 IMU、机体速度、23 个编码器位置与速度、上一拍 23 维动作，以及由赛场视觉系统提供的球和对手相对位置/速度、局部进攻方向和回合进度。Actor 不读取世界绝对坐标、真实接触力、换权真值或犯规真值。

Critic 为训练期的 190 维集中价值网络，在同一局部观测之外读取攻击者、足球和双方世界状态、真实触球归属、碰撞力、清球、控球和安全状态。部署时丢弃 Critic，不会把特权量泄漏给机器人端 Actor。

## 当前真实证据

| 证据 | 结果 | 能证明什么 |
|---|---:|---|
| `metrics/v048_t1_23dof_contract_audit_run_02.json` | 通过 | 主项目 23 DoF 合同未被破坏，安全检查使用正式 RCSS 机械力矩上限 |
| `metrics/v048_semantic_audit_run_07.json` | 通过 | 两台精确 RCSS T1、23 维动作、93/190 维观测；精确球/球场接触参数、真实攻击者首触→防守者触球换权→0.325 米清球链，以及真实跨机器人碰撞/上身犯规传感器均可工作 |
| `metrics/v048_cpu_smoke_run_01.json` | 1 环境 × 8 步通过 | 单世界有限状态 stepping |
| `metrics/v048_cpu_smoke_run_03.json` | 16 环境 × 8 步通过 | 当前代码下 32 台 T1、16 个足球的 MJLab 多世界向量化 stepping |
| `metrics/v048_cpu_ppo_training_smoke_run_02.json` | 1 次 PPO 更新通过 | 当前代码下 93→23 Actor、190→1 Critic、采样和反向传播接线完整 |

碰撞审计和 smoke 都是基础设施证据，不是抢断能力成绩。当前没有正式 GPU 长训、没有候选检查点、没有固定 20 回合验证、没有 100 个 held-out 回合，也没有录像，因此 `quality_passed=false`，不得晋级比赛或团队决策。

## 固定与封存合同

正式门槛在 `configs/v048_competitive_acceptance.json`：固定集为 20 个种子；封存集为从 4810000 开始的 100 个未见种子，并要求至少 8 个方向区间和 5 个速度区间。两套测试都要求成功率至少 65%、犯规率不高于 2%、摔倒率不高于 2%，每次成功都必须同时满足真实换权、0.25 米清球、2 秒控球和所有安全门禁。

## 复现

在正式 WSL 仓库 `/home/linh/robot_soccer/training_runs/rcss_t1_training_20260706` 中执行：

```bash
export PYTHONPATH="$PWD:/home/linh/robot_soccer/third_party/mjlab/src:/home/linh/robot_soccer/third_party/mjlab_playground:${PYTHONPATH:-}"
/home/linh/robot_soccer/envs/mjlab-playground/bin/python experiments/v048_fullbody_legal_tackle/scripts/audit_v048_semantics.py
/home/linh/robot_soccer/envs/mjlab-playground/bin/python experiments/v048_fullbody_legal_tackle/scripts/smoke_v048.py --steps 8 --num-envs 16 --seed 48001
```

训练入口已经接通 GPU 并行，但本轮按审计要求没有启动 GPU 长训：

```bash
cd /home/linh/robot_soccer/third_party/mjlab_playground
export PYTHONPATH="/home/linh/robot_soccer/training_runs/rcss_t1_training_20260706:/home/linh/robot_soccer/third_party/mjlab/src:$PWD:${PYTHONPATH:-}"
/home/linh/robot_soccer/envs/mjlab-playground/bin/python /home/linh/robot_soccer/training_runs/rcss_t1_training_20260706/experiments/v048_fullbody_legal_tackle/scripts/train_v048.py --num-envs 4096 --iterations 3000 --seed 48000 --nan-guard
```

下一阶段不能直接宣布完成。应先把已经验收的步态/起身策略作为低层初始化接入双方，再训练攻击者策略池，之后才进行 GPU 长训、固定验证、封存测试和视频审计。
