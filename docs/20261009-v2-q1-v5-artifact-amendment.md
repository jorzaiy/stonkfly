# V2-Q1 v5 artifact amendment

v5 的所有动力学、compatibility、observer 和 full-graph gates 均通过，但
raw NPZ 未单独保存 micro/compatibility trace，manifest 也没有 clean-state
生成记录。为使正式产物满足预注册的 artifact schema，补齐 raw trace 和
clean checkpoint 后使用新 v6 目录补跑；v5 保留不覆盖。
