# v042 门将站位与地滚球封堵

本课程让1号门将保持面向来球，使用已验证步态横移，并把一只脚而不是身体中心放到球路上。射门是4.2–4.8 m/s、目标在门心左右0.55米内的直线地滚球；成功必须有门将真实接触，随后球反向或降速且未进球。

## 当前实测

| 证据 | 结果 |
|---|---|
| `metrics/v042_smoke_8_steps.json` | 8步安全冒烟通过 |
| `metrics/v042_policy_fixed_eval.json` | 512步PPO候选8/10封堵、1/10失球、0摔倒 |
| `metrics/v042_policy_held_out_test.json` | 20个未见种子16/20封堵、4/20失球、0摔倒 |
| `metrics/v042_policy_iter_0002_training.json` | 从 iter_0001 续训256步，总步数768，未覆盖旧模型 |
| `metrics/v042_policy_fixed_eval_run_02.json` | 续训候选仍为8/10，没有证明优于 iter_0001 |

这只通过了“窄范围直线地滚球站立封堵”的本地第一关，不能称为完整门将。正式合同还要求至少300球的角度×速度×物理扰动分格测试、扑救率不低于70%，并完成出击、倒地扑救、起身、控球后安全出球和后卫冲突检查。当前底层没有扑倒或起身策略，门将也不能把球稳定传给队友，因此 `v042` 不得晋级正式3v3或7v7。

## 2026-08-20 严格门槛复审

旧计时器在门将碰球后，如果球再次朝球门危险运动，会保留此前累计的安全时间；不同安全片段理论上可能被拼成0.3秒。现已改为只计算连续安全帧，危险运动立即清零。旧 `quality_passed` 需要用新逻辑复核；同一 `iter_0001` 的严格文件 `metrics/v042_policy_fixed_eval_run_05.json` 仍为8/10、失球1/10、0摔倒，`stage_quality_passed=true`，但完整门将 `quality_passed=false`。20个未见种子的对应证据为 `metrics/v042_policy_held_out_test_run_03.json`，同样只通过窄范围直线地滚球第一关。

```bash
python experiments/v042_goalkeeper/scripts/smoke_v042.py
python experiments/v042_goalkeeper/scripts/train_v042.py --timesteps 512
python experiments/v042_goalkeeper/scripts/train_v042.py --timesteps 256 --resume experiments/v042_goalkeeper/checkpoints/v042_policy_iter_0001.zip
python experiments/v042_goalkeeper/scripts/evaluate_v042.py --model experiments/v042_goalkeeper/checkpoints/v042_policy_iter_0001.zip
python experiments/v042_goalkeeper/scripts/evaluate_v042.py --model experiments/v042_goalkeeper/checkpoints/v042_policy_iter_0001.zip --held-out
```
