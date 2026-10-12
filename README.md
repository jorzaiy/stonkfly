![Stonkfly: a pixel fly beside a candlestick chart](assets/stonkfly.png)

# Stonkfly Orchard

一个保留 MaleCNS 果蝇连接组结构、探索简化蘑菇体神经计算的开源项目。当前主角不再参加交易或学习考试，而是在二维果园里自由活动。

> **我们的 fly**<br>
> 前量化交易员，现居果园。<br>
> 性格孤僻，喜欢花，不爱香蕉，偶尔去休息区发呆。<br>
> 对人类安排的奖励机制持保留态度。

## 当前可以玩什么

Stonkfly Orchard 是一个本地网页演示：香蕉、花朵和休息区持续产生合成气味，fly 的方向同时受到神经偏好、随机探索和环境物理影响。页面会显示位置、轨迹、附近气味、KC/APL 活动、连续 MBON 偏好及记忆变化。

神经转向、随机探索和物理规则在代码与界面中明确分开；没有隐藏 waypoint，也不会在后台替 fly 选择香蕉。接触香蕉只会向既有工程模型注入一次明确记录的人工 reward DAN，它依然可以不买账。

### 本地启动

需要 Python 3.11 和已准备好的 MaleCNS 数据：

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m stonkfly prepare
python -m stonkfly.orchard.server --seed 20261012
```

打开 `http://127.0.0.1:8765`。页面支持启动、暂停、单步和相同 seed 重放。换一个非负整数 seed 会同时改变三种合成气味的 PN 模板与随机探索序列。

通过 Tailscale 查看时，推荐只绑定本机的 Tailscale 地址：

```sh
python -m stonkfly.orchard.server \
  --host 你的Tailscale地址 \
  --port 8765 \
  --seed 42
```

没有 MaleCNS 数据时，可用明确标记的 UI fixture：

```sh
python -m stonkfly.orchard.server --toy --seed 42
```

默认日志写入被 Git 忽略的 `runs/orchard/`。完整架构、代理假设和控制方程见 [Orchard 说明](docs/STONKFLY_ORCHARD_ARCHITECTURE.md)。

## 研究历程

| 路线 | 状态 | 结论 |
|---|---|---|
| MaleCNS v1 金融学习 | 已关闭并归档 | 没有找到稳定、稀疏且可用于后续学习验证的神经工作区间；历史执行代码仅为复现保留。见 [最终归档](docs/20261010-male-cns-v1-final-archive.md)。 |
| Mushroom Body Lab v2 | 正式 NO-GO，探索结果保留 | 合成气味表征稳定且可分；工程可塑性产生了可重复的配对差分更新，但没有可靠跨过固定零阈值。见 [v2 实验汇总和轻量数据](docs/STONKFLY_V2_EXPERIMENT_SUMMARY.md)。 |
| Stonkfly Orchard | 可运行原型 | 不再设置学习考试；观察固定模型如何感知、漫游和改变连续记忆。 |

这些结果不证明真实果蝇采用相同动力学、学习率或行为读出，也不证明盈利学习。人工 DAN 信号不是疼痛、快乐或意识模型。

## 代码位置

- `stonkfly/orchard/`：果园环境、E1-G/M2 神经适配器与本地服务器
- `stonkfly/mb_lab/`：合成气味动力学、连接组拓扑和工程可塑性
- `experiments/mb_lab/`：M1/M2 实验 runner
- `docs/results/stonkfly_v2/`：可公开复核的轻量结果数据
- `tests/orchard/`、`tests/mb_lab/`：果园与蘑菇体测试

```sh
PYTHONPATH=. python -m pytest -q tests/orchard tests/mb_lab
```

## 历史交易代码

旧的 Coinbase 纸面/实盘执行与 A 股研究代码仍留在仓库用于历史复现，但不是 Orchard 的依赖，也不是当前默认入口。任何实盘行为都需要用户自行配置凭证并明确启用；项目不会随果园启动交易。操作限制与恢复说明见 [operations.md](docs/operations.md)，模型边界见 [model.md](docs/model.md)。
