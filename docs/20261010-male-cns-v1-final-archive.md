# Stonkfly / MaleCNS v1 最终归档总结

状态：**2026-10-10 归档。MaleCNS v1 的金融学习与交易实验正式结束。本文是本路线的最终权威入口；旧文档和失败目录继续作为证据保留，但不再作为继续训练的授权。**

## 1. 最终结论

本项目没有获得“Fly 学会炒股”的证据。

在冻结的完整 MaleCNS v1.0 图、当前通用 LIF/电流型突触、统一 `contact × 0.275` 权重映射和已实现的感觉代理下，视觉与嗅觉路线都没有形成学习所需的工作区间：KC 表征不是沉寂，就是进入大范围、持续且无法退出的高活动状态。视觉 graded 路线还出现了经严格数值资格审查确认的线性失稳。

因此，早期训练或模拟盘中出现的动作、决策差异和盈亏差异，不能归因于网络学到了市场规律。按冻结的 O2 停止条件，sensory adapter、plasticity、S3、paper-market training 和所有真实执行均不再继续。

这是对本项目工程模型的结论，不是对真实果蝇、真实蘑菇体或生物神经系统学习能力的结论。

## 2. 我们实际测试的系统

- 完整保留 MaleCNS v1.0 图，不通过删节点、删边或隐藏选股来制造结果；
- 166,700 个节点、25,582,938 条聚合边、124,177,617 个突触接触；
- 4,064 个 KC；
- 市场观察、感觉代理、神经传播、可塑性、固定解码和执行限制分层；
- reward/P&L 只能作为对已标识 dopamine cells 的工程输入，不代表痛苦、快乐或意识；
- 正式动力学诊断不读取市场、收益、动作或交易标签；
- 长跑保留预注册、runner/test hashes、raw、formal、report、manifest、launch 和 stdout；失败与不完整目录不覆盖。

这些边界使最终否定结果具有可审计性，也限制了结论的适用范围。

## 3. 证据链

### 3.1 早期训练与 S2：行为差异不等于学习

早期训练曾产生动作与盈亏差异，但后续追踪发现，生产运行 609 个事件中有 607 个处在同一持续上态。KC 活动主要反映网络是否已被点燃，而不是行情刺激内容。

S2–S2j 对视觉输入到 KC 的传播、时间窗、固定向量和 Jaccard 进行了逐层检查。原先看似“中等重叠”的汇总被证明会掩盖低态/高态混合；系统表现为离散翻转和强路径依赖，没有稳定的刺激特异稀疏码。因此 S3 和收益训练被阻塞。

S2k–S2n 的反事实与因果诊断说明：KC→KC 递归兴奋并不是唯一原因，增强 APL 或窗口化削弱局部输入可以改变幅度，却不能消除翻态、历史依赖或下游静默。不能靠一个局部倍率把现象解释成合格表征。

### 3.2 R0/R0b/R0c：吸收上态与动力学资格问题

R0 的合成剂量–反应正式实验完成了 252 个刺激记录。生产臂在低态和约 1,400 KC 的高态之间硬切换；APL×2 反事实臂能把部分高峰压到 111–267 KC，却仍非单调、有历史依赖，并令目标 MBON07/11 全静默。冻结判定为 inconclusive，而不是“接近成功”。

R0b 进一步确认，一旦进入高态，在无新刺激条件下仍持续发放；这更符合吸收式上态，而不是刺激编码。R0c 的 `dt` 审计没有提供一个足以恢复当前模拟器动力学解释资格的稳定工作区间。

这一阶段也推翻了“只要继续扫 APL、阈值或输入强度就能找到好点”的做法：在缺少外部生物依据时，按 KC 输出或收益选择这些参数属于结果后适配。

### 3.3 V2 视觉路线：沉寂与线性失稳

P0 对完整图进行了守恒与分层盘点。Q1 建立并通过了隔离 hybrid kernel 的实现资格审查，这说明后续失败不能简单归因于 runner 没有工作。

视觉路线随后得到两个互相补强的负结果：

- C0 单侧 graded release 截断负向 deviation，KC 始终沉寂，正式 verdict 为 `C0_NO_QUALIFYING_GAIN`；
- Q2 双向 graded release 恢复负向传递后，严格 protocol-repair 通过 A1–A7、dense reference、复模态和 Hessenberg locks，正式 verdict 为 `Q2_LINEAR_INSTABILITY`。两个起点均得到多个正实部 Ritz 值，最大 `Re(s)` 约 `0.05679147 ms^-1`。

因此视觉路线于 2026-10-10 正式关闭。事后 SCC 分析把主模态解释为 lamina 高增益负反馈振荡；这是支持性机制解释，不是删边或选择缩放参数的授权。

### 3.4 O0–O2 嗅觉路线：最终停止实验

O0 使用固定结构规则识别出 290 个合格 cholinergic direct PN，其中固定哈希抽取的 8-PN panel 覆盖了可审计的 PN→KC 原图通路；O0 判定为 `O0_PN_PROXY_INVENTORY_QUALIFIED`。

O1 在完整图上分别提交单个 engineered PN spike-event：

- 7 个 PN 的事件到达了直接目标和 KC，但没有触发 spike；
- `DM4_adPN` 的一个事件在三个 `dt` 下都触发约 132 万次全脑 spike，最后 500 ms 仍约 33 万次，queue、conductance 和 adaptation 均未归零；
- 实现、传播、重复确定性和跨 `dt` 分类通过，稳定性失败；
- verdict 为 `O1_SINGLE_EVENT_INSTABILITY`。

O2 是 O1 停线后的 `counterfactual: true` 最终诊断，不是放宽 O1。它保留每个 PN 的全部原始出边、符号和相对比例，只按源节点出边 L1 归一化总剂量。294 个 trials 覆盖 8 PN、6 个固定剂量、3 个 `dt` 和 2 次重复：

| L1 dose | 结果 |
|---:|---|
| 128 | 全部 KC 沉寂 |
| 256 | 全部 KC 沉寂 |
| 512 | 全部 KC 沉寂 |
| 1024 | 7 个沉寂，`M_lPNm11D` 持续失稳 |
| 2048 | 5 个沉寂，3 个持续失稳 |
| 4096 | `DL1_adPN` 仍无 KC spike，其余 7 个持续失稳 |

没有任何 PN/dose 组合属于 `stable_nonzero`。失稳组合在前 50 ms 激活约 2,850–3,200 个 KC（约 70–79%），全脑约 66 万次发放，最后 500 ms 仍约 33 万次。高剂量下不同 PN 的早期 KC 集合高度重叠（两两 Jaccard 平均约 0.956），说明输入差异大体被共同高活动吸引子淹没。

O2 的实现、input ledger、294-trial 数量、重复确定性、跨 `dt` 分类、finite/bounds、ZERO controls 和 start/end hashes 全部通过，但 `candidate_doses=[]`。正式 verdict：

`O2_NO_STABLE_NONZERO_BAND_CLOSE_V1`

## 4. 可以与不可以声称什么

### 可以声称

1. 当前完整图与当前 neuron/synapse/weight mapping 的组合，在预注册测试范围内没有展示稳定、稀疏、刺激特异、跨 `dt` 一致且可退出的 KC 工作区间。
2. 更换视觉/嗅觉入口没有解决共同的沉寂—失稳问题。
3. 当前表征不具备进入可塑性或交易学习实验的最低资格。
4. 继续按神经输出或收益细扫剂量、阈值、抑制或局部倍率，既违反停止规则，也缺少独立识别依据。

### 不可以声称

1. 不可以声称真实果蝇不能学习、没有意识，或真实神经系统具有相同失稳。
2. 不可以声称实验数学上排除了所有连续剂量中的极窄稳定区间；O2 检验的是冻结六档剂量。但继续追点不再属于 v1。
3. 不可以把“MB 外网络先被引燃”写成 O2 已完成的因果证明；O2 的时间分辨率只支持它作为机制推断。
4. 不可以声称蘑菇体计算能力严格不超过逻辑回归。当前证据只说明本实现尚未展示相对简单基线的优势。
5. 不可以把任何历史回测、动作或盈亏差异称为已验证的 profitable learning。

## 5. 最终范围决定

- MaleCNS v1 金融学习/交易路线：**关闭并归档**；
- 视觉 Q2 与嗅觉 O 系列：**不再追加剂量点、机制旋钮或补跑**；
- sensory adapter 集成、plasticity、S3 和 paper-market training：**停止**；
- 真实交易：**从未由这些实验授权，继续禁止**；
- 现有代码、数据、失败目录和正式产物：**保留，不覆盖，不删除**。

以后若重启相关研究，必须另立模型版本和章程，不能把它称为 v1 的继续训练。

## 6. 权威文档索引

| 文档 | 作用 | SHA256 |
|---|---|---|
| `docs/20261006-s2-diagnosis-for-opus.md` | S1–S2j 权威诊断 | `3c117dc1b13943ebcc4a5bb7944b97bbbc4601843f59fe1e8a16ae46145459f7` |
| `docs/20261007-r0-results-for-opus.md` | R0/R0b 结果与审阅 | `fae16a8946299356976397f2925c429d4a81b2ac2526e4386e6658a6bd69b2f0` |
| `docs/20261008-model-scope-and-stop-review.md` | V2 模型范围与停止条件 | `47396caeca7b4b57c0c1db4e12c1e053d774901e95a3dec3e2c20090ce42284e` |
| `docs/20261010-q2-visual-route-stop-scope.md` | 视觉路线关闭 | `dfa853f64051ea7ea5af4987602d46933efe03ba13381f1a75671789d80ee6dc` |
| `docs/20261010-o0-olfactory-pn-inventory-preregistered.md` | O0 预注册 | `8c9718d98c7661148386549d78d9a9c37e8e825f512790662022fa7213db74fb` |
| `docs/20261010-o1-olfactory-pn-single-event-preregistered.md` | O1 预注册 | `35374da9e620bdcec2212f1438426f7c518b9b1fdb0784b9eacf4b8b91912922` |
| `docs/20261010-o2-pn-normalized-dose-diagnosis-preregistered.md` | O2 冻结停止实验 | `cd864ec9e1fc1250b638bef5df8159e796970fe7319f2e972af68bd292440488` |
| `docs/20261010-o2-review-and-project-direction-opus.md` | O2 独立审阅与方向意见 | `a2dd22ccd5abb0cba82c4bc77c0714783122ebdcb4214aaefdc7c1b77f7293aa` |

## 7. 关键正式结果索引

| 阶段 | 正式结果 | SHA256 | 判定 |
|---|---|---|---|
| R0 | `runs/plasticity_diagnosis/R0/20261007T-synthetic-dose-response-v2/r0_formal.json` | `d5f294f5c84aebfafa22a3539315791a56827a76e907ebadcfd8743e809c4e21` | inconclusive，回到设计评审 |
| R0b | `runs/plasticity_diagnosis/R0b/20261007T-absorbing-state-v1/r0b_formal.json` | `5f40904f57098802d28733f1b7929d5de5a0a8254969cf9900a32dc43f71c38b` | 吸收上态证据 |
| R0c | `runs/plasticity_diagnosis/R0c/20261008T-dt-convergence-v3/r0c_formal.json` | `b531533ae1a401afebeae5317e4a1ef6b963e1afedd1ad2018dd071ecc0dec51` | `dt` 资格审计 |
| P0 | `runs/whole_brain_v2/P0/20261008T-model-inventory-v3/v2_p0_formal.json` | `7fa88023cc09a12955174ab76f8cebb5a05ddf9c258867511187440bad69b45f` | 完整图分层盘点 |
| Q1 | `runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v7/q1_formal.json` | `394c2cd05ecc54c979b9d871176a86bb82670c850c108fe91c63ba8354ead574` | `Q1_IMPLEMENTATION_QUALIFIED` |
| C0 | `runs/whole_brain_v2/C0/20261010T-graded-release-calibration-v3/c0_formal.json` | `6dd290e4758453bdc3bf52c61bd2f28b9161b8aeffdf56db4838a6c00c3` | `C0_NO_QUALIFYING_GAIN` |
| Q2 | `runs/whole_brain_v2/Q2/20261010T-bidirectional-graded-kernel-v2-protocol-repair/q2_formal.json` | `c97ebade9075545ad3ee97d7992c2478687a4516d19904a695206925c0bab971` | `Q2_LINEAR_INSTABILITY` |
| O0 | `runs/whole_brain_olfactory/O0/20261010T-pn-proxy-inventory-v1/o0_formal.json` | `8c09ccfccc9946bf95892cb1c77aa80ce60331e79394deb0ebb9f5372602f69b` | `O0_PN_PROXY_INVENTORY_QUALIFIED` |
| O1 | `runs/whole_brain_olfactory/O1/20261010T-pn-single-event-v1/o1_formal.json` | `8b19cd27314cdfeb1870c6307c7aac4ee1c41e5ccc86871b1ee0ae910e3cdb08` | `O1_SINGLE_EVENT_INSTABILITY` |
| O2 | `runs/whole_brain_olfactory/O2/20261010T-pn-normalized-dose-diagnosis-v1/o2_formal.json` | `4331ca5c70bc8c80aab359114f5830844b8e5c6b379a4de3ccd2e9c5d4de37ec` | `O2_NO_STABLE_NONZERO_BAND_CLOSE_V1` |

O2 最终产物 hashes：

`runs/`、market data、logs、ledger 和 checkpoints 按仓库安全规则仅保留在本地，不进入公开 GitHub。公开归档提供协议、runner、测试、结论与内容哈希；下表用于对本地正式产物验真。

| 产物 | SHA256 |
|---|---|
| `o2_raw.npz` | `d33f51e3f91a9ac9dad4b19d562f1d3bb3360aa141bbff909978fd8ce9c441a6` |
| `o2_formal.json` | `4331ca5c70bc8c80aab359114f5830844b8e5c6b379a4de3ccd2e9c5d4de37ec` |
| `o2_report.md` | `1582526bbb3f6d107645862e5a7cf1a3386e4215652c648d3cf80343fa06e03d` |
| `o2_manifest.json` | `ca1f1c0c1ff3e9987f0700b80d037a81d35dea036efb7b9f73b6385c6c5a8037` |

## 8. 如果未来再开一个好玩的项目

### 归档提交验证状态

公开归档前完整测试套件共收集 433 项：415 passed、4 skipped、5 failed、9 errors。9 个 errors 来自未公开的外部历史输入目录；2 个旧 2021 preflight 被当前数据质量门 fail-closed；另外 3 个旧 V5/linked-reinforcement 测试因后续源码与历史 authorization/source hashes 不再一致而 fail-closed。没有为了得到全绿结果而回写历史授权哈希。R0/V2/O0–O2 当前动力学链另行执行 68 项独立测试，全部通过。

如果目标是交易效果，应先建立简单、便宜、可解释的常规量化基线，并把样本外、成本、泄漏和基线比较写在任何复杂模型之前。

如果目标是神经科学风格的研究或展示，应另立新项目，并在运行前决定：

1. 是否仍保留完整连接组，还是建立有生物依据的 PN→KC→APL→MBON/DAN 子系统；
2. neuron/synapse 参数来自哪些外部生物数据，而不是来自“调到稳定”或“调到赚钱”；
3. 合成刺激下的表征门：稀疏、刺激特异、跨 `dt` 一致、可退出；
4. 市场接入后的基线门：固定特征下必须在样本外优于逻辑回归、随机策略和买入持有；
5. 明确的计算预算与停止条件，失败后不追加机制。

MaleCNS v1 没有学会炒股，但它留下了一套比“偶然赚到钱”更可靠的东西：面对一个很可爱的想法，我们没有偷偷替它选答案，而是一步步确认它真正做了什么，并在证据要求停止时停下。

项目在这里归档。以后找到新的好玩想法，再从一张干净的预注册开始。
