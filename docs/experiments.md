# 实验记录

> 每个 run 在合并 PR 前登记到本文件。只记录结论和关键指标，完整日志留在 `outputs/`。

## 统一协议

- 划分：`prompt_id` grouped CV，禁止按行随机切分；统一使用 `ntl.data.grouped_cv_splits`。
- 共享 fold 文件：`python scripts/make_folds.py`（默认 5 折、seed=0）。
- OOF：验证折覆盖全部训练行；主指标按 prompt 抽 1–3 条计算，模拟隐藏测试。
- 指标：balanced accuracy，附每类召回率。
- 所有对比必须使用同一 fold、同一 seed、同一测试形态。

## 记录

| 日期 | run_id | 成员 | 路线 | 方法 / 主要改动 | 验证协议 | balanced acc | 对比基线 | 结论 / 下一步 |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |
