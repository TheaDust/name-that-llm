# 共享脚本

已提供以下入口：

| 脚本 | 作用 |
|---|---|
| `make_folds.py` | 生成统一的 prompt-grouped CV 划分，缓存到 `data/processed/folds.parquet` |
| `prepare_open_pool.py` | 清洗公开 open pool，写出 `data/processed/open_pool_clean.parquet` 与报告 |
| `check_data.py` | 检查数据目录、schema、fold 对齐和 open pool 是否就绪 |
| `make_submission.py` | 读取 `test_proba.parquet`，argmax 后校验并写出 `submissions/submission.parquet` |
| `validate_submission.py` | 校验列名、id 覆盖、空值、标签合法性 |

用法：

```powershell
python scripts/make_folds.py
python scripts/prepare_open_pool.py
python scripts/check_data.py
python scripts/validate_submission.py --submission submissions/submission.parquet
python scripts/make_submission.py `
    --proba outputs/ensemble/test_proba.parquet `
    --out submissions/submission.parquet
```

约定：

- `folds.parquet` 由一人生成并分发，各路线用 `grouped_cv_splits(train, folds)` 读取同一划分。

- 脚本只做编排，核心逻辑放在 `src/ntl/`。
- 所有脚本支持 `--help`，路径通过参数传入，不硬编码本机路径。
- 不把数据、权重、密钥写进脚本。
