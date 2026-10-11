# V2-Q1 v2 runner infrastructure amendment

日期：2026-10-09

正式 v2 目录 `runs/whole_brain_v2/Q1/20261009T-hybrid-kernel-qualification-v2/`
已按预注册要求启动，但在任何 full-graph trial 开始前停止。runner 将
`build()` 函数对象误当作其返回的 build manifest 记录，抛出
`TypeError: 'function' object is not subscriptable`，因此 v2 目录只保留
启动记录、stdout 和 formal.log，不含伪造的正式 verdict。

该错误是实现基础设施错误，不是动力学或 compatibility 结果。修复后只能
使用新版本 v3 目录补跑；v2 目录不得覆盖或就地重跑成功。
