# Fly 神经网络训练实验总结 (2026-10-04)

> **历史文档，不再作为下一步依据。** S2–S2j 已将当前首要阻塞定位为感觉输入到 KC 的离散阈值/吸引子动力学，不应继续用收益扫塑性参数。请以 [20261006-s2-diagnosis-for-opus.md](20261006-s2-diagnosis-for-opus.md) 为当前权威交接。

## 实验目标

调整训练参数使 reward/aversive 比例更平衡，改善 MBON 输出和交易信号质量。

## 背景问题

之前的训练存在以下问题：
- **stimulus 比例失衡**: aversive 60.5% vs reward 39.5%
- **突触变化单向**: 43% 突触被抑制，仅 2.7% 增强
- **MBON 结构性不对称**: MBON11 总权重是 MBON07 的 1.82x（connectome 固有特性）

## 参数调整

修改 `experiments/a_share/replay_wc006.py:609-615`:

| 参数 | 旧值 | 新值 | 目的 |
|------|------|------|------|
| `absolute_loss_weight` | 1.5 | 1.0 | 减少 aversive 强度 |
| `absolute_gain_weight` | 1.0 | 2.0 | 增强 reward 强度 |
| `absolute_gain_threshold` | 0.005 | 0.002 | 降低 reward 触发门槛 |

## 训练结果

### Stimulus 分布改善

| 指标 | 旧训练 | 新训练 | 变化 |
|------|--------|--------|------|
| reward | 42.0% (256天) | 41.4% (252天) | -0.6% |
| aversive | 54.2% (330天) | 48.4% (295天) | **-5.8%** |
| none | 3.8% | 10.2% | +6.4% |

aversive 比例从 54.2% 降至 48.4%，有所改善但仍高于 reward。

### 突触变化仍然失衡

从 clean brain 到训练后的权重变化：
- **总变化边数**: 3,566 / 25,582,938
- **增强**: 214 edges (6.0%)
- **抑制**: 3,352 edges (94.0%)

尽管 stimulus 比例改善，突触变化仍然是 **94% 抑制 vs 6% 增强**。

## 前向测试结果

### MBON 激活对比

| 测试 | MBON07 mean_hz | MBON11 mean_hz | 总信号 | BUY | SELL |
|------|----------------|----------------|--------|-----|------|
| 无预训练 baseline | 1.17 | 17.15 | 477 | 0 | 477 |
| 新训练 balanced-reward-v2 | **0.00** | 0.03 | 11 | 0 | 11 |
| 旧训练 low-threshold-0.5 | **0.00** | 0.03 | 14 | 0 | 14 |

### 关键发现

1. **MBON07 完全沉默**: 预训练后 MBON07 的 mean_hz = 0.00，无法产生 BUY 信号
2. **MBON11 大幅抑制**: 从无预训练的 17.15 Hz 降至 0.03 Hz
3. **无 BUY 信号**: 所有预训练模型都只产生 SELL 信号

## 问题根因分析

```
训练期间:
  - reward 触发时: 增强 KC→MBON07 连接 (approach)
  - aversive 触发时: 抑制 KC→MBON11 连接 (avoidance)
  
实际效果:
  - 94% 突触被抑制 → 两个 MBON 都被大幅削弱
  - MBON07 本身连接数少 (3651 edges vs MBON11 4184 edges)
  - MBON07 更容易被抑制到沉默
```

塑性规则可能存在不对称性，或者 aversive 信号的抑制效果传播到了 MBON07 相关的 KC 连接。

## 输出文件

- **训练输出**: `runs/wc006_purged_fly_training/20261004T-balanced-reward-v2/`
  - `brain-day-1.npz`: 最终训练 brain checkpoint
  - `events.jsonl`: 609 天训练事件记录
  - `provenance.json`: 训练配置和元数据

- **前向测试**: `runs/fly_purged_heldout_2024_2025/20261004T-balanced-reward-v2/`
  - 483 天 heldout 测试 (2024-01-02 至 2025-12-29)
  - 11 个 SELL 信号，0 个 BUY 信号

## 下一步建议

### 方案 A: 进一步调整塑性参数
- 大幅增加 `absolute_gain_weight` (如 5.0 或更高)
- 降低 `absolute_loss_weight` (如 0.5)
- 可能需要多轮实验找到平衡点

### 方案 B: 检查塑性规则实现
- 验证 reward/aversive 是否正确作用于目标 MBON
- 检查是否存在跨 MBON 的抑制传播
- 可能需要修改 centered rate rule 的实现

### 方案 C: 结构性解决方案
- 为 MBON07 和 MBON11 使用不同的塑性系数
- 考虑 MBON 权重的归一化或平衡机制
- 探索 connectome 结构性不对称的补偿方法

### 方案 D: 替代解码策略
- 使用绝对激活值而非 MBON07-MBON11 差值
- 尝试不同的 threshold 策略
- 探索其他神经元群作为信号源

## 相关文件

- `stonkfly/neural/brain.py:436-466`: restore_weights_only() 实现
- `experiments/a_share/replay_wc006.py:609-615`: 塑性参数配置
- `experiments/a_share/run_purged_fly_training.py`: 训练脚本入口
- `experiments/a_share/run_purged_fly_base_2024_2025.py`: 前向测试脚本
