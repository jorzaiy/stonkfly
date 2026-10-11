# O0 olfactory-PN proxy inventory（冻结预注册）

状态：**2026-10-10 冻结；授权只读执行 O0。O0 是 Q2/visual route 关闭后的独立项目，不运行神经动力学，不启用 learning、reward、decoder、market 或 execution。**

## 1. 目标

验证 MaleCNS v1.0 原图中是否存在一个可按固定声明字段与图关系识别、规模足够、直接连接 KC 的 antennal-lobe PN proxy 候选集合，为后续 O1 单 spike-event 安全资格提供固定输入。

O0 不证明真实嗅觉编码，不证明 KC 功能，不选择输入强度，也不授权训练。

## 2. 固定输入

| 输入 | SHA256 |
|---|---|
| `data/graph.npz` | `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b` |
| `data/normalized/neurons.feather` | `2103b5203d2ba5502c06fb70a4e0b23edcbcf79ed0c344722101941fc289ba2d` |
| `data/normalized/edges.arrow` | `842b762c9427c794eaedd453178aa034727a5084850a91fdcc4d3fffe09f5b7e` |
| Q2/visual stop-scope | 在 O0 启动前记录实际 SHA256 |

必须重新确认 166,700 nodes、25,582,938 directed edges、124,177,617 contacts，graph IDs 与 neuron rows 完全对齐。不得剪图、筛边或改权重。

## 3. 冻结集合规则

所有匹配均基于原始声明字段和原图，不读取任何动力学或收益结果。

1. `ORN`：`cell_type` 以 `ORN_` 开头；
2. `ORN-direct PN`：至少收到一条 ORN 原图边，`cell_type` 包含大小写敏感字符串 `PN`，且 `superclass == cb_intrinsic`；
3. `KC`：`cell_type` 以 `KC` 开头；
4. `eligible excitatory PN proxy`：同时满足 ORN-direct PN、`neurotransmitter == acetylcholine`，并至少有一条直接原图出边到 KC。

不按 edge weight、KC coverage 排名、cell activity、MBON、动作或收益选择 eligible 节点。GABA PN 只做描述，不进入首个 proxy 集合。

预检查已经看到了集合规模，O0 不是盲测；已知预期为 ORN 2,635、ORN-direct PN 594、eligible excitatory PN 290。正式 O0 的价值是固定完整 ID/edge/hash ledger，而不是把这些已知计数包装成发现。

## 4. O1 固定 probe panel 生成规则

仅当 O0 gates 通过，生成八节点 `O1 probe panel`：

- 对每个 eligible PN 的 unsigned 64-bit `source_id` 取 little-endian 8 bytes 的 SHA256；
- 按 `(digest bytes, source_id)` 升序排序；
- 取前 8 个节点；
- 不查看其 edge weight、KC target 数或任何动力学结果；
- 保存 node index、source ID、cell type、neurotransmitter、individual digest 和 panel hash。

该 panel 只用于 O1 安全/传播 probe，不是未来 market feature mapping。正式 feature→PN mapping 必须在 O1 通过后另写预注册。

## 5. 只读 ledger

O0 保存：

- 四个集合的 node count、node-index/source-ID SHA256；
- ORN→PN 与 eligible PN→KC 的 edge count、weight sign/count/sum/absolute sum、first/last edge order hash；
- eligible PN 的 cell-type、source ID、ORN incoming edge count、KC outgoing edge count；
- eligible PN 覆盖的唯一 KC count 与 KC ID hash；
- GABA ORN-direct PN 的相同描述摘要；
- graph scale、ID alignment、finite weights 与 contacts conservation；
- O1 panel 与生成规则 hash。

## 6. Gates 与判定

只允许：

1. `O0_IMPLEMENTATION_FAILURE`：pin、图完整性、ID 对齐、集合互斥/规则复算、hash stability 或产物缺失；
2. `O0_NO_QUALIFYING_PN_SET`：eligible PN 少于 32、直接 KC edges 为零或唯一 KC coverage 少于 1,000；
3. `O0_PN_PROXY_INVENTORY_QUALIFIED`。

计数与预检查不一致本身不自动调规则；应先判 implementation failure 并解释数据/代码差异。

## 7. 产物与后续

正式目录：

`runs/whole_brain_olfactory/O0/20261010T-pn-proxy-inventory-v1/`

至少包含 `o0_raw.npz`、`o0_formal.json`、`o0_report.md`、`o0_manifest.json`。禁止覆盖失败目录。

若 O0 通过，下一步另写 O1 single-event qualification：向 panel 中每个 PN 分别提交恰好一个、显式记录的 engineered spike-event intent，测试完整图传播、确定性、dt、撤刺激退出和吸收上态。O1 不得直接刺激 KC/MBON，也不得以 KC 点燃作为安全资格的通过条件。
