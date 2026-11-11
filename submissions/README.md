# 最终提交文件

`scripts/make_submission.py`生成的 `submission.parquet` 放在这里。

约定：

- 文件名固定为 `submission.parquet`，Kaggle Notebook 直接取用。
- 本地保留历史版本时可命名为 `submission_YYYYMMDD.parquet`。
- 该文件不被 Git 跟踪；本 README 除外。
- 提交前必须通过 `scripts/validate_submission.py`的校验：
  - 两列 `id`, `model`；
  - `id` 与 `test.csv` 完全一致且唯一；
  - 无空值；
  - `model` 只包含 18 个合法标签。
