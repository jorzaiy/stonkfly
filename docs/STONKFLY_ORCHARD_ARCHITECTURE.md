# Stonkfly Orchard 最小原型架构

日期：2026-10-12 UTC
状态：本地探索性演示；不是 M1/M2 补考，不改变任何历史 verdict。

## 1. 目标与边界

Orchard 让一只由简化蘑菇体输出影响的虚拟 fly 在二维果园中持续活动。果园包含香蕉、花朵和休息区，页面展示轨迹、局部气味、KC/APL 活动、连续 MBON 偏好与记忆变化。

它不是行为学模型：没有实现真实果蝇运动回路、触角采样、风场、饥饿神经调制或导航神经环。接触香蕉产生的 reward DAN 是人工工程输入。项目不读取旧 RT，不接入市场、交易、P&L 或执行模块。

## 2. 四层职责

1. `stonkfly.orchard.core.OrchardSimulation`：二维气味场、能量、速度、边界反射和区域接触。物理层只能落实运动结果，不能替 fly 选择目标。
2. `stonkfly.orchard.brain.MaleCNSOrchardBrain`：用固定参数运行三种 Orchard 气味的 E1-G PN→KC→APL 响应；以真实 KC→MBON07/11 边和 M2 `PlasticityState` 计算连续偏好与记忆。
3. 显式控制混合：神经偏好乘以各气味空间梯度得到 `neural_turn`；固定 seed 的有色随机过程产生 `exploration_turn`；总转向严格为 `neural_gain*neural_turn + exploration_gain*exploration_turn`，随后才进入物理层。日志逐项保存，不存在隐藏 waypoint 或“看见目标就转向”的脚本。
4. `stonkfly.orchard.server` 与原生 Canvas 页面：本地 HTTP 控制、状态轮询、绘图和简洁 JSONL 日志；网页不计算或修改神经决策。

## 3. 感觉与学习复用

- 参数保持 E1-G：`tau_k=10 ms`、`tau_apl=50 ms`、`input_scale=1`、`apl_gain=1`、`threshold=0.35`。
- 交互演示采用 E1 已测试过的 `dt=0.1 ms`，启动时对香蕉、花朵、休息区三个确定性 29/145 PN 模板各计算一次，并缓存 KC mean rate 与 APL 活动。
- 运行时用局部气味浓度线性混合缓存表征。这是为了实时性的代理，不等同于对混合气味重新积分完整 E1 非线性方程。
- KC→MBON 使用真实右侧连接、固定归一化 readout 和现有 M2 10 ms 可塑性状态；`eta=0.1 s⁻¹` 保持不变。首次进入香蕉区域会触发一次现有 `train_event` 的完整气味—reward DAN 配对时间线，离开区域后才重新武装；其余世界步只推进感觉、trace 与记忆状态。
- 连续偏好由基线连接产生的中心化 innate 分量，加上当前相对基线的 learned valence 构成。该固定 decoder 是工程控制接口，不是生物学行为读出。

## 4. 世界与控制

气味浓度为以各区域中心为源的固定高斯场。正偏好沿梯度靠近该气味，负偏好沿梯度远离。随机探索是独立 seeded RNG 状态；重置会同时恢复位置、heading、RNG 与可塑权重，所以相同 seed 和相同控制日程可重放。

休息区只通过公开物理规则恢复能量；不向控制器偷偷注入转向。能量仅缩放速度。边界采用反射。每个状态快照都包含 neural/exploration/total turn、气味浓度、区域接触与脑状态。

## 5. 本地运行

```sh
PYTHONPATH=. .venv/bin/python -m stonkfly.orchard.server --seed 20261012
```

打开 `http://127.0.0.1:8765`。服务初始暂停，可启动、暂停、单步或用相同 seed 重置。默认会在被 Git 忽略的 `runs/orchard/` 写入 `session.json` 和 `events.jsonl`；传入 `--no-log` 可关闭日志。

没有本地 MaleCNS 数据时，可用快速 UI fixture：

```sh
PYTHONPATH=. .venv/bin/python -m stonkfly.orchard.server --toy --seed 20261012
```

`--toy` 只验证交互路径，页面会明确显示 `toy-fixture`，不得冒充 MaleCNS 神经结果。

## 6. 最小验收与后续

本阶段测试固定 seed 重放、边界物理、控制贡献分离、香蕉 reward 对 toy memory 的影响、暂停/单步命令、日志追加和静态页面资源。另做一次真实 MaleCNS 启动 smoke，确认 E1-G/M2 适配器可产生有限状态。

后续可增加风向、多个 fruit 实例、可调时间倍率和保存/加载 replay，但不能把这些环境规则包装为神经决策。任何真实行为学解释需另行加入运动回路与独立验证；Orchard 本身不以 PASS/FAIL 为目标。
