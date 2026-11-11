# 处理后数据（不提交到 Git）

用于存放跨 run 复用的中间产物，例如：

```text
data/processed/
├── folds.parquet              # 共享的 prompt-grouped CV 划分
├── qwen3_embedding_8b/        # B 的层特征缓存
└── features/                  # 其他可复用的稀疏/稠密特征
```

约定：

- 与某一次实验强绑定的产物放 `outputs/<route>/<run_id>/`，不放这里。
- 本目录整体被 `.gitignore` 忽略。
- 缓存文件名带上模型、层、pooling、数据版本，避免不同实验互相覆盖。
