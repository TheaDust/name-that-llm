# 实验记录

> 每个 run 在合并 PR 前登记到本文件。只记录结论和关键指标，完整日志留在 `outputs/`。

## 统一协议

- 划分：`prompt_id` grouped CV，禁止按行随机切分；统一使用 `ntl.data.grouped_cv_splits(train, folds)`，其中 `folds` 从共享文件读取。
- 共享 fold 文件：`python scripts/make_folds.py`（默认 5 折、seed=0），由一人生成并分发，各 run 保存同一文件的副本。
- OOF：验证折覆盖全部训练行，主指标在完整 OOF 上统一计算。
- 指标：balanced accuracy，附每类召回率。
- 所有对比必须使用同一 fold 文件；训练随机种子记录在各 run 配置中。
- 使用外部 open pool 时，记录清洗参数、采样 ID、采样量、seed 和 `pool_weight_ratio`；采样后按当前训练折重算权重；pool 只进训练折，不参与验证。

## 记录

| 日期 | run_id | 成员 | 路线 | 方法 / 主要改动 | 验证协议 | balanced acc | 对比基线 | 结论 / 下一步 |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |
