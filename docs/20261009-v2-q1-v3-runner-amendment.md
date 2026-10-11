# V2-Q1 v3 runner infrastructure amendment

v3 补跑在任何 full-graph trial 开始前停止。v2 修复了 build manifest 的
首次错误，但 `source_end`/`binary_end` 仍残留对 `build` 函数名的索引，导致
相同的 `TypeError`。v2 与 v3 目录均保留原始启动和 traceback；修复后必须
使用新的 v4 目录，不得覆盖 v2/v3。
