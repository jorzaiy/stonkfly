# V2-Q1 v4 implementation amendment

v4 完成了 24 个完整图 trial 及全部微回路/compatibility gate，但正式
verdict 为 `Q1_NUMERICAL_FAILURE`：observer 主状态等价和重复确定性失败。
审计定位为 `HybridBrain.active` 未使用槽位由 `np.empty` 留下未初始化字节，
它们被纳入规范化状态哈希；动力学门限、解析方程、compatibility、bounds、
queue、fine-dt 和权重冻结均通过。修复为全零初始化后只能使用新的 v5 目录，
不得覆盖 v4。
