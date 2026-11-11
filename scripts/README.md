# 共享脚本

已提供以下入口：

| 脚本 | 作用 |
|---|---|
| `make_folds.py` | 生成统一的 prompt-grouped CV 划分，缓存到 `data/processed/folds.parquet` |
| `make_submission.py` | 读取 `test_proba.parquet`，argmax 后校验并写出 `submissions/submission.parquet` |
| `validate_submission.py` | 校验列名、id 覆盖、空值、标签合法性 |

用法：

```powershell
python scripts/make_folds.py
python scripts/validate_submission.py --submission submissions/submission.parquet
python scripts/make_submission.py `
    --proba outputs/ensemble/test_proba.parquet `
    --out submissions/submission.parquet
```

约定：

- 脚本只做编排，核心逻辑放在 `src/ntl/`。
- 所有脚本支持 `--help`，路径通过参数传入，不硬编码本机路径。
- 不把数据、权重、密钥写进脚本。
