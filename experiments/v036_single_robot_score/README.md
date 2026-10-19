# v036 单机器人追球射门

目标：先不做 7v7，让一台 Booster T1 在真实 RCSSServerMJ 七人制球场中追球，并把球射入右侧真实球门。

训练分两步：先用 v033 已验收追球逻辑生成真实 MuJoCo 示范，对三维 Actor 做行为克隆；再用 PPO 根据接近球、球向球门推进、进球、站立和跌倒奖励继续优化。Actor 输出的是前进、横移、转向三个动作；23关节控制仍由官方 `T1.pt` 与 `RcssT1Controller` 完成。

```bash
python experiments/v036_single_robot_score/scripts/evaluate_v036.py --teacher
python experiments/v036_single_robot_score/scripts/train_v036.py --timesteps 4096
python experiments/v036_single_robot_score/scripts/evaluate_v036.py
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python experiments/v036_single_robot_score/scripts/record_v036.py
```

历史检查点、指标都使用递增编号保存，不覆盖以前的证据。
