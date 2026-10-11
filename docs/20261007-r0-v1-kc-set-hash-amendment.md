# R0 v1 产物编码勘误：KC set hash 的索引口径

状态：**已冻结勘误；不改变 R0 的刺激、参数、运行顺序、完整图约束或判读规则。**

## 发现的问题

R0 v1 runner 将 `kc_set_hash` 计算为 active KC 的 MaleCNS body ID 集合哈希。冻结的 S2i `digest_ids` 实际接收的是 authoritative circuit mask 中的 KC 图内索引（`kc[active]`），不是 body ID。两者都是同一 active KC 集合的编码，但哈希输入不同，因此不能直接比较。

因此，R0 v1 的失败只属于产物编码/桥接门禁失败，不是神经计算、刺激构造或模型结果失败。v1 已留下 `r0_formal_failure.json`，不得作为正式结果使用。

## 证据

对生产配置 C 的五个固定锚点 `uniform_20/40/80/160/255`，逐项核对结果为：

- frame SHA256：全部一致；
- active-KC count：全部一致；
- full counts/spike SHA256：全部一致；
- v1 body-ID hash：20 恰好相同，其余四个因编码口径不同而不相同；
- 用 authoritative KC 图内索引重新计算后，五个 `kc_set_hash` 均与 S2i `digest_ids` 等价。

固定证据文件：

- 预注册：`docs/20261007-r0-synthetic-dose-response-preregistered.md`
  - SHA256 `d613471ea81943a904083587711853a0a92adec990b87bd45c65d6c84de60c64`
- S2i calibration：`runs/plasticity_diagnosis/S2i/20261005T-synthetic-calibration-grid-v1/calibration_grid.json`
  - SHA256 `5f843d9ee6e3aae6d47102669b6b5189a304807b0763fae994f3f0472aea81ef`
- v1 runner（勘误前）：SHA256 `216b614cf5be742939af9ac8085fe07c7f9e04e91a050909993037c585ae4958`

## 修正

v2 runner 将：

- 正式 `kc_set_hash` 统一改为 S2i 的 authoritative KC 图内索引口径；
- 若记录 body ID 集合，单独命名为 `kc_body_id_set_hash`，不得与 `kc_set_hash` 混用；
- 新增非空 KC 集合回归测试，直接验证图内索引哈希与 S2i `digest_ids` 等价；
- 保留五锚点的 frame/count/full spike hash 桥接门禁。

这份 amendment 只解释并修正编码口径，不授权添加刺激点、改变参数、重写判读阈值或修改冻结预注册正文。
