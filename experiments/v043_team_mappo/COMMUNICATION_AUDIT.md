# v043 分散执行与通信边界

v043 在一个 Python 进程中批量推进多台机器人，这是仿真加速方式，不等于球员之间共享信息。比赛部署路径必须满足以下边界。

- 共享 Actor 对每名球员逐行执行相同的线性层、LayerNorm 和 GRUCell；网络没有跨球员注意力、池化或拼接。
- 每名球员维护自己的 128 维 GRU 状态；重置某名球员的历史不需要读取其他球员的隐藏状态。
- Actor 输入只有本体状态、局部视野内的球/队友/对手、公开球场方向、球衣号和本地推断角色。不可见实体槽位为零且 mask 为零。
- 全局 `BallEventTracker` 用于奖励、指标与集中式 Critic。Actor 只获得“自己当前/最近是否触球”，不会收到全队最后触球者或全局球权转换广播。
- 动态角色有两套：`assign_decentralized_roles` 根据公开场地区域、球衣号和本机可见球确定压迫权，是比赛 Actor 使用的局部推断；`assign_dynamic_roles` 是使用全局真值的训练教师，只能生成课程示范或对照动作。
- 96维全局状态只传给训练期 `central_critic`。部署调用 `deterministic_step(local_observation, own_hidden)`，不调用 Critic。
- `networks.py`、`observations.py` 和 `roles.py` 的部署路径不导入 socket、HTTP、进程间消息或共享内存通信库。

自动审计脚本 `scripts/audit_decentralized_execution.py` 会检查：改变其他球员观测行不会改变目标球员动作；禁用 Critic 后 Actor 仍能推理；全局触球事件不会广播给未触球球员；改变视野外队友不会改变本地压迫角色；部署路径不存在 IPC/网络导入。

集中训练仍然使用仿真全局真值，这是 CTDE 的训练特权，不代表比赛时允许通信。任何未来新增的跨球员注意力、全局角色 one-hot、全局球权或传球消息都必须重新经过规则审计，不能直接接入 Actor。
