# v037 从中场连续控球到射门

这个版本不把球瞬移到门前，而是严格执行五级距离课程：门前、禁区内、禁区外、进攻三区、中场。Actor 仍只输出前进、横移和转向；官方 `T1.pt` 负责腿部动作，完整23关节由 `RcssT1Controller` 发送。

新增的控球奖励会同时衡量：球是否向右侧球门缩短距离、球是否回到球门中心线、触球后机器人是否持续处于0.9米控制范围、是否丢球、是否跌倒以及是否由官方裁判判进球。

每一级只有最新固定种子评估通过，下一阶段才允许启动：

```bash
python experiments/v037_midfield_dribble_score/scripts/train_stage.py --stage 0 --timesteps 8192
python experiments/v037_midfield_dribble_score/scripts/train_stage.py --stage 1 --timesteps 8192
python experiments/v037_midfield_dribble_score/scripts/train_stage.py --stage 2 --timesteps 8192
python experiments/v037_midfield_dribble_score/scripts/train_stage.py --stage 3 --timesteps 8192
python experiments/v037_midfield_dribble_score/scripts/train_stage.py --stage 4 --timesteps 8192
```

直接动作版在禁区内阶段出现了“第一次触球后忘记继续追球”的退化，因此保留其失败指标，后续改用 r2 教师残差版。r2 的几何教师负责绕到球后方和基本追球，PPO只输出小幅速度/横移/转向修正：

```bash
python experiments/v037_midfield_dribble_score/scripts/train_residual_stage.py --stage 0 --timesteps 4096
python experiments/v037_midfield_dribble_score/scripts/train_residual_stage.py --stage 1 --timesteps 4096
python experiments/v037_midfield_dribble_score/scripts/train_residual_stage.py --stage 2 --timesteps 4096
python experiments/v037_midfield_dribble_score/scripts/train_residual_stage.py --stage 3 --timesteps 4096
python experiments/v037_midfield_dribble_score/scripts/train_residual_stage.py --stage 4 --timesteps 4096
```

最终模型的独立评估和完整中场录像：

```bash
python experiments/v037_midfield_dribble_score/scripts/evaluate_residual_stage.py --stage 4
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python experiments/v037_midfield_dribble_score/scripts/record_midfield_goal.py
```

最终 `v037r2_stage_04_iter_0002.zip` 从距离球门 26–27 米的中场区域开始，在5个独立固定种子中由官方裁判判进4球，成功率80%；平均推进24.49米，平均受控持球比例94.01%，无跌倒、无丢球，23路关节目标全部正常发送。Windows 可直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_from_windows_vscode.ps1 v037-status
powershell -ExecutionPolicy Bypass -File scripts/run_from_windows_vscode.ps1 v037-evaluate-midfield
powershell -ExecutionPolicy Bypass -File scripts/run_from_windows_vscode.ps1 v037-record-midfield
powershell -ExecutionPolicy Bypass -File scripts/run_from_windows_vscode.ps1 v037-continue-midfield
```
