# RCSSServerMJ 球权与裁判接口审计

当前安装的 `rcsssmj 0.2.1` 提供了足够可靠的原生信号来构造球类技能奖励：

- `SoccerBall.active_contact`：当前正在与球接触的 `AgentID`。
- `SoccerBall.active_contact_body_parts`：当前接触涉及的脚、小腿、手等几何体。
- `SoccerBall.last_contact`：当前接触释放或更换后保存的前一位球员。
- `SoccerBall.contact_change`：接触归属最后变化的仿真时间。
- `SoccerBall.get_most_recent_contact()`：裁判处理出界和角球时也使用的最近触球方。
- `SoccerReferee`：依据真实接触判定开球触发、二次触球、手球、角球、球门球和进球；1号门将在本方禁区内有手触球例外。

因此新课程不再使用“球移动超过4厘米就算某人触球”的旧近似。v040的 `BallEventTracker` 使用 `active_contact` 与逐帧接触部位；同一球员保持接触但部位改变时会记录 `body_parts_change`，从而避免“脚接触变成手接触”漏检。v043的团队跟踪器另外读取 `last_contact` 与 `contact_change`。A释放后B接触才会记录球权转移。

仍需注意两个边界：第一，`active_contact` 同一物理帧只能给出一个当前球员，多人同时夹球时必须结合事件序列处理；第二，裁判的犯规日志没有直接暴露成训练环境字段，正式抢断压力测试还需要额外记录比赛模式变化和被罚下状态。
