# 测试

当前已覆盖以下契约，防止三条路线在合并时出现静默错误：

1. **数据 schema**：`train.csv` 列名、18 个标签、`prompt_id`/模型组合唯一性。
2. **指标**：共享评估报告的 balanced accuracy 与 `sklearn.metrics.balanced_accuracy_score` 一致。
3. **防泄漏**：共享 fold 按 id 对齐；拒绝数据不匹配和同一 prompt 跨折。
4. **概率契约**：`oof_proba` / `test_proba` 的列名、id 唯一性、行和为 1。
5. **提交校验**：缺列、重复 id、空值、非法标签都能被拒绝。

测试文件命名为 `test_*.py`，运行：

```powershell
pip install -e ".[dev]"
pytest -q
```
