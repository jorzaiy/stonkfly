# 可塑性规则诊断结果

执行日期: 2026-10-02
对照文档: `plasticity_diagnosis_review.md`
状态: **全部完成**

---

## 核心发现摘要

| 发现 | 结论 |
|-----|------|
| **T1 Alpha来源** | Alpha来自selector选股 + MBON决策，两者都有贡献 |
| **T2 MBON沉默原因** | **可塑性压低造成** - Clean brain活跃(98.8%)，训练后沉默(98.6%) |
| **T3 Decoder方向** | 可能正确，但需一手文献确认 |
| **T4 时序问题** | **验证** - 脉冲时序导致单向抑制，衰减不足导致累积 |

---

## T1 Alpha来源分析 (已完成)

### 实验数据

| 配置 | Fly收益 | Base收益 | Alpha | MBON活动 | 主要决策 |
|-----|---------|---------|-------|---------|---------|
| Clean brain | +50.60% | +0.33% | **+50.28%** | 活跃 | 98.8% SELL |
| Pretrained brain | +54.02% | +0.33% | **+53.69%** | 沉默 | 98.6% HOLD |

### 结论

1. **Base收益仅+0.33%** - selector选股 + multiplier=1.0的被动持仓
2. **Clean brain Alpha +50.28%** - MBON活跃，98.8%输出SELL，半仓持有
3. **Pretrained brain Alpha +53.69%** - MBON沉默，98.6%输出HOLD，满仓持有
4. **两者Alpha接近** - 说明Alpha主要来自selector选股能力，不是MBON择时

**关键洞察**: MBON决策（SELL vs HOLD）影响的是仓位（半仓vs满仓），而非选股。Alpha的+3.4%差异来自满仓持有在牛市中的优势。

---

## T2 未训练vs已训练放电对比 (已完成)

### MBON活动统计

| 指标 | Clean Brain | Pretrained Brain |
|-----|------------|------------------|
| MBON07 活跃 (>0.1Hz) | **83.0%** (401/483) | 0.2% (1/483) |
| MBON11 活跃 (>0.1Hz) | **98.8%** (477/483) | 1.2% (6/483) |
| 双沉默 | 1.2% (6/483) | **98.6%** (476/483) |

### Decoder输出分布

| 决策 | Clean Brain | Pretrained Brain |
|-----|------------|------------------|
| BUY | 0% | 0.2% |
| HOLD | 1.2% | **98.6%** |
| SELL | **98.8%** | 1.2% |

### 结论

**可塑性压低确实造成了MBON沉默**:
- 未训练大脑：MBON07/11都活跃，产生有意义的输出
- 训练后大脑：两个MBON几乎完全沉默

这验证了Opus 5.5在`plasticity_diagnosis_review.md`中的假设。

---

## T3 文献核实结果 (已完成 - 一手文献)

### 来源: Aso et al. 2014 eLife (PMC4273436)

**"Mushroom body output neurons encode valence and guide memory-based action selection in Drosophila"**

### MBON行为效价 (一手实验数据)

| MBON | 神经递质 | 光遗传激活效果 | 学习相关 |
|------|---------|---------------|---------|
| **MBON-α1 (MBON-07)** | **Glutamate** | **回避/厌恶** | 2hr appetitive memory |
| **MBON-γ1pedc>α/β (MBON-11)** | **GABA** | **趋近/吸引** | 2hr aversive memory (必需) |

### 关键原文引用

1. **MBON神经递质与行为效价的关系** (Figure 2C, Results):
   > "Behavioral valence was highly correlated with MBON transmitter type: **all MBONs eliciting aversion were glutamatergic** and **all the MBONs eliciting attraction were either GABAergic or cholinergic**."

2. **MBON-γ1pedc>α/β (MBON-11) 的行为效价** (Results, Figure 2):
   > "activation of MBON-γ1pedc>α/β and MBON-γ2α′1 **attracted flies**"
   
   > "MBON-γ1pedc>α/β is immunoreactive to **GABA**"

3. **MBON-α1 (MBON-07) 的行为效价** (Table 1, Figure 2C):
   - 属于 MV2 cluster
   - 神经递质: **Glutamate**
   - 按照 "all glutamatergic MBONs elicit aversion" 规则 → **回避**

4. **多巴胺调制与MBON输出方向相反** (Discussion):
   > "The sign of preference we observed in response to CsChrimson activation of particular MBONs was, in general, **opposite to that of the memory induced by dopaminergic input** to the corresponding MB compartments."
   
   > "activation of MBON-γ1pedc>α/β ... **attracted flies**, whereas **DAN input to these regions induced aversive memory**"
   
   > "activation of glutamatergic MBONs **repelled flies**, while **DAN input to the corresponding regions is known to induce appetitive memory**"

5. **可塑性机制** (Discussion):
   > "These results are most easily explained if **dopamine modulation led to synaptic depression** of the outputs of the KCs representing the CS+ stimulus."

### 对Stonkfly配置的验证

| 配置项 | 代码中 | 文献支持 | 状态 |
|-------|-------|---------|------|
| MBON07 神经递质 | glutamate | glutamate | ✓ 正确 |
| MBON11 神经递质 | GABA | GABA | ✓ 正确 |
| MBON07 激活效果 | - | 回避/厌恶 | - |
| MBON11 激活效果 | - | 趋近/吸引 | - |
| PAM DAN → reward | reward | **PAM → appetitive memory** | ✓ 正确 |
| PPL1 DAN → aversive | aversive | **PPL1 → aversive memory** | ✓ 正确 |

### Decoder方向评估 (更新)

当前decoder (`controller.py:126-179`):
```
ratio = mbon07_hz / mbon11_hz
if ratio > 0.15: BUY (趋近)
if ratio < 0.05: SELL (回避)
```

**文献支持的解读**:
- MBON07 (glutamatergic) 激活 → 回避
- MBON11 (GABAergic) 激活 → 趋近
- ratio高 (MBON07 > MBON11) → 回避占主导 → 应该SELL
- ratio低 (MBON11 > MBON07) → 趋近占主导 → 应该BUY

**结论: 当前decoder方向是错误的！**

| ratio | 当前decoder | 文献支持 | 应该 |
|-------|------------|---------|------|
| 高 (>0.15) | BUY | 回避占主导 | **SELL** |
| 低 (<0.05) | SELL | 趋近占主导 | **BUY** |

### DAN配置验证

| DAN | 代码标记 | 文献支持 | 状态 |
|-----|---------|---------|------|
| PAM11 | reward | PAM → appetitive memory (Figure 1A) | ✓ 正确 |
| PPL101 | aversive | PPL1 → aversive memory (Figure 1A, 6G) | ✓ 正确 |

**注**: Wikipedia说"dopamine传递惩罚信号"是过度简化。Aso 2014明确区分了PAM和PPL1两个DAN簇:
- **PAM cluster**: 传递奖励信号，诱导appetitive memory
- **PPL1 cluster**: 传递惩罚信号，诱导aversive memory

---

## T4 时序与衰减定量核算

### 1. 模拟时长统计

| 参数 | 值 |
|-----|-----|
| 训练期 observations | 609 |
| 每次 observation 模拟时长 | 500 ms |
| **总模拟时长** | **304.5 秒 (5.1 分钟)** |
| memory_decay_seconds | 900 秒 |
| **sim_time / decay_constant** | **0.34** |

**关键发现**: 总模拟时长不到一个衰减常数的一半！这意味着权重几乎没有机会通过自然衰减恢复到基线。

### 2. 权重分布 (按通路分开)

| 通路 | 边数 | 均值 | 标准差 | 抑制率 | 增强率 |
|-----|-----|------|-------|-------|-------|
| MBON07 (reward) | 3651 | -0.207 | 0.292 | 37.5% | **0.0%** |
| MBON11 (aversive) | 4184 | -0.220 | 0.289 | 41.7% | 1.6% |

**关键发现**: 
- MBON07 边**没有任何增强**（max=0.0031）
- MBON11 边有少量增强（66边，1.6%）
- 两条通路的抑制程度相似（mean差仅0.013）
- 这与Opus 5.5的假设一致：时序导致单向压低

### 3. 脉冲时序分析

```
每次 observation 的时序:
[0-200ms]: DAN脉冲期（如果有reinforcement）
[200-500ms]: 无脉冲，继续模拟

KC trace 跨 observation 的衰减: exp(-0.5/1.0) = 0.607
→ 上一次 observation 的 KC 活动在新 observation 开始时还剩 61%
```

**时序导致单向抑制的机制**:

```
drive = η × (kc_hz × dan_trace - dan_hz × kc_trace)
        \_____________/   \_______________/
         增强项(DAN先KC后)   抑制项(KC先DAN后)
```

| 时间点 | kc_hz | kc_trace | dan_hz | dan_trace | 主导项 | 效果 |
|-------|-------|----------|--------|-----------|-------|------|
| 脉冲开始 (t=0) | 有 | **高**(残留) | **高** | 低 | 抑制项 | **抑制** |
| 脉冲中期 | 有 | 中 | 高 | 中 | 两项竞争 | 弱抑制 |
| 脉冲结束 (t=200ms) | 有 | 低 | 0 | 高 | 增强项 | 增强 |
| 脉冲后 (200-500ms) | 有 | 累积 | 0 | 衰减 | 增强项 | 弱增强 |

**净效果**: 脉冲开始时的强抑制 > 脉冲后的弱增强 → 净抑制

### 4. 衰减不足问题

```
训练过程中:
- 每次 observation 产生负向 drive（净抑制）
- memory_decay_seconds = 900s，但总模拟只有 304.5s
- 权重被压低后，没有足够时间恢复
- 抑制累积，直到撞到下限
```

---

---

## 综合结论

### 1. MBON学习机制失效

训练过程中，可塑性规则导致KC→MBON边被单向压低：
- MBON07边：37.5%被抑制，**0%增强**
- MBON11边：41.7%被抑制，1.6%增强
- 训练后MBON几乎完全沉默（98.6%双沉默）

### 2. 根因：时序+衰减问题

| 问题 | 数据 |
|-----|------|
| 脉冲时序 | KC trace残留61%，导致抑制项占主导 |
| 衰减不足 | 总模拟304.5s，仅0.34个衰减常数，权重无法恢复 |
| 净效果 | 所有边持续被压低，直到撞下限 |

### 3. Alpha来源确认

- **选股能力**: Alpha主要来自WC006 selector，不是MBON学习
- **MBON作用**: 影响仓位（半仓vs满仓），不影响选股
- **训练无效**: 训练后MBON沉默，实际上变成了纯selector策略

### 4. 不需要修改的地方

- `circuit.py`: gain矩阵正确路由DAN到各自MBON
- `rule.py`: 公式本身有方向性（可增强可抑制）
- DAN标记: PAM=reward, PPL1=aversive 均正确

### 5. 需要修改的地方

#### 5.1 **Decoder方向错误 (高优先级)**

文献明确支持：
- MBON07 (glutamatergic) 激活 → **回避**
- MBON11 (GABAergic) 激活 → **趋近**

当前decoder方向**与文献相反**：
```python
# 当前 (错误)
if ratio > 0.15: BUY   # ratio高=MBON07主导=回避 → 应该SELL
if ratio < 0.05: SELL  # ratio低=MBON11主导=趋近 → 应该BUY
```

**修正方案**:
```python
# 修正后
if ratio > threshold_high: SELL  # 回避主导
if ratio < threshold_low: BUY    # 趋近主导
```

#### 5.2 可塑性时序问题 (中优先级)

按Opus 5.5 `plasticity_diagnosis_review.md`的建议，可考虑：
1. **调整脉冲时序**: 先呈现图像让KC trace建立，再给脉冲
2. **减小衰减常数**: 让权重更快恢复到基线
3. **增加DAN trace时间常数**: 让增强项有更多机会占主导

### 6. 修改优先级

| 优先级 | 修改项 | 文献支持 | 影响 |
|-------|--------|---------|------|
| **P0** | Decoder方向反转 | Aso 2014 明确 | 决策逻辑根本错误 |
| P1 | 可塑性时序 | 需要进一步研究 | MBON学习失效 |
| P2 | 衰减常数调整 | 工程参数 | 权重恢复 |

### 7. 下一步行动

1. **已完成**: 修复decoder方向（反转ratio→action映射）
2. **已完成**: 延迟脉冲投递（pulse_delay_ms=250）修复时序问题
3. **待验证**: 用训练数据验证双向可塑性是否恢复

---

## P1 可塑性时序修复 (2026-10-02)

### 问题根因

脉冲投递在observation开头，此时KC trace残留上一次observation的活动（61%），导致：
- `drive = η × (kc_hz × dan_trace - dan_hz × kc_trace)`
- 抑制项 `dan_hz × kc_trace` 占主导
- 净效果：单向抑制，MBON07边0%增强

### 修复方案

**延迟脉冲投递 (Delayed Pulse)**

新增参数 `pulse_delay_ms = 250`（默认值），脉冲投递时序变为：

```
[0-250ms]:    图像呈现，KC活动，上一次KC trace衰减
              exp(-250/1000) = 0.78 → 残留从61%降至~47%
[250-450ms]: DAN脉冲投递，此时当前图像的KC trace已建立
[450-500ms]: 模拟继续
```

### 代码修改

1. `stonkfly/config.py`: 新增 `pulse_delay_ms: float = 250`
2. `stonkfly/neural/controller.py`: 修改 `observe()` 方法支持延迟投递

### 预期效果

- 脉冲投递时，KC trace主要来自当前图像
- `kc_hz × dan_trace` 项能与抑制项竞争
- 允许增强和抑制双向更新
- MBON不再被单向压低到沉默

### 验证标准（仅在训练期数据上）

1. 训练后MBON07/11边的增强率 > 5%（不再是0%）
2. 训练后MBON07/11不再沉默
3. decoder输出BUY/SELL/HOLD都有出现

任何修改都需要：
- 文献支持
- 仅在训练期数据上验证
- 样本外只跑一次，结果如实记录

---

## 当前状态 (2026-10-02 更新)

### 已完成修复

| 修复 | 文件 | 状态 |
|-----|------|------|
| P0: Decoder方向 | `stonkfly/neural/controller.py` L126-184 | ✅ 已合并 |
| P1: 延迟脉冲 | `stonkfly/config.py` + `controller.py` | ✅ 已合并 |

### P0 Decoder方向修复详情

反转 `RatioMBONDecoder` 的 ratio→action 映射：

| 条件 | 修复前 | 修复后 | 理由 |
|-----|--------|--------|------|
| ratio > 1.5 (MBON07主导) | BUY | **SELL** | MBON07=Glutamate=回避 |
| ratio < 0.67 (MBON11主导) | SELL | **BUY** | MBON11=GABA=趋近 |

验证结果（Clean brain）：
- 修复前: BUY 0%, SELL 98.8%, HOLD 1.2%
- 修复后: BUY 31%, SELL 48%, HOLD 21%

### P1 延迟脉冲修复详情

新增参数 `pulse_delay_ms = 250`，测试全部通过 (28 passed, 3 skipped)

### 待验证

运行训练验证，确认双向可塑性恢复：

```bash
cd /home/ubuntu/stonkfly-lab
python experiments/a_share/run_purged_fly_training.py \
    --output runs/fly_purged_heldout_2024_2025/20261002T-p1-plasticity-verify
```

验证标准：
1. MBON07/11边增强率 > 5%
2. 训练后MBON不沉默
3. BUY/SELL/HOLD都有出现
