# V2-P0 v1/v2 纯实现失败修正与门禁语义澄清

状态：**2026-10-08 冻结；在读取任何正式 P0 结构结果之前写入。** 原预注册 `docs/20261008-v2-p0-model-inventory-preregistered.md` 的模型范围、输入、0.20 符号灵敏度触发线、6-hop 上限和 95% 覆盖线均不变。

## 1. 保留的失败

### v1

`runs/whole_brain_v2/P0/20261008T-model-inventory-v1/` 以 `P0_IMPLEMENTATION_FAILURE` 停止。runner 错误地假设 `data/normalized/edges.arrow` 已处于 CSR 行顺序；`prepare.py` 实际先以稳定排序按 `pre_index` 构造 CSR。v1 没有产生结构结果。

修正：按 `prepare.py` 的 `np.argsort(pre_index, kind="stable")` 复现顺序后，再对 source、post 和 contact 做全量一致性检查。不得按 post、contact 或结果重新排序。

### v2

`runs/whole_brain_v2/P0/20261008T-model-inventory-v2/` 完成结构计算后，在组装正式 JSON 时以 `KeyError('comparison')` 停止。原因是 runner 把 R0c 判定误读为顶层 `comparison.verdict`；正式文件中的路径是 `metrics.comparison.verdict`。v2 没有写出正式结构 JSON/NPZ，现有 failure JSON 和日志必须保留。

修正：只修复冻结历史 JSON 的字段映射：

- R0：正式判定固定为原预注册要求的 `inconclusive`；
- R0b：`metrics_and_hypotheses.verdict`；
- R0c：`metrics.comparison.verdict`；
- R0c 的各 dt 指标继续从 `metrics.by_dt` 原样复制。

## 2. 95% 与 KC→MBON 门禁语义

原预注册中的“graded candidates 在 6 hops 内可到达至少 95% 的 KC”固定解释为三个互斥 graded candidate 主声明层的**并集**作为多源起点，计算一次最短有向路径覆盖。三组仍必须分别非空并分别报告 hop 分布，但不要求每一组单独覆盖 95% KC。

原预注册中的“KC 经 retained paths 在 6 hops 内可到达 MBON07 与 MBON11”固定解释为以**全部 KC 的并集**为多源起点，分别检查两种 MBON 目标至少有一个可达节点；不得限定为先由某一 graded 子组到达的 KC 子集。

这是对集合量词的实现前澄清，不改变 6-hop、95% 或目标类型门槛，也不允许根据结果选择 union/per-group 版本。per-group 和 graded-reached-KC 子集结果仍可作为描述性统计，不参与正式资格判定。

## 3. v3 额外实现检查

- 正式运行前增加 history schema 单元测试，固定上述三条 JSON 路径；
- 比例门禁必须真的验证所有名为 fraction/proportion 的值有限且位于 `[0,1]`，不能只检查 finite；
- 正式退出码必须同时受确定性、源文件不变和全部实现门禁控制；
- 正式目录改为 `runs/whole_brain_v2/P0/20261008T-model-inventory-v3/`；
- v1/v2 不覆盖、不删除、不补写成功文件。

除此之外不得改变分类优先级、边/接触守恒、符号规则、路径算法、历史结论或资格标签。
