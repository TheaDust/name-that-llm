# 测试

当前已覆盖以下契约，防止三条路线在合并时出现静默错误：

1. **数据 schema**：`train.csv` 列名、18 个标签、`prompt_id` 分组完整性。
2. **指标**：自定义 balanced accuracy 与 `sklearn.metrics.balanced_accuracy_score` 一致。
3. **防泄漏**：同一 `prompt_id` 不会同时出现在训练折和验证折。
4. **概率契约**：`oof_proba` / `test_proba` 的列名、id 唯一性、行和为 1。
5. **提交校验**：缺列、重复 id、空值、非法标签都能被拒绝。

测试文件命名为 `test_*.py`，运行：

```powershell
pip install -e ".[dev]"
pytest -q
```
