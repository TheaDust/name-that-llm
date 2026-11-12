# 处理后数据（不提交到 Git）

用于存放跨 run 复用的中间产物：

```text
data/processed/
├── folds.parquet              # 共享的 prompt-grouped CV 划分
├── open_pool_clean.parquet    # 清洗后的公开 open pool
├── open_pool_clean.json       # open pool 清洗报告
├── qwen3_embedding_8b/        # B 的层特征缓存
└── features/                  # 其他可复用的稀疏/稠密特征
```

## 生成

```powershell
python scripts/make_folds.py          # -> folds.parquet
python scripts/prepare_open_pool.py   # -> open_pool_clean.parquet / .json
```

## folds.parquet

列：`id, prompt_id, model, fold`。

- 固定 5 折，按 `prompt_id` 分组，同一 prompt 不跨折。
- 每轮 4 折训练、1 折验证，5 轮得到完整 OOF。
- 使用 `ntl.data.grouped_cv_splits(train, folds)` 读取。
- 不要按行随机切分，也不要重新随机生成。

## 验证

```powershell
python scripts/check_data.py
python scripts/check_data.py --skip-pool   # 不使用 open pool 时
```

## 分发

`folds.parquet` 和 `open_pool_clean.parquet` 可以直接分发给队友，队友不需要重跑脚本。分发后比对 SHA-256：

```powershell
Get-FileHash data/processed/folds.parquet, data/processed/open_pool_clean.parquet -Algorithm SHA256
```

## 约定

- 与某一次实验强绑定的产物放 `outputs/<route>/<run_id>/`，不放这里。
- 本目录整体被 `.gitignore` 忽略。
- 缓存文件名带上模型、层、pooling、数据版本，避免不同实验互相覆盖。
- `folds.parquet` 必须与全队使用同一份 `train.csv` 对应。

## 训练权重

`open_pool_clean.parquet` 保存清洗样本和来源。完成采样或过滤后，每个训练折调用
`ntl.open_pool.pool_sample_weights(pool, train_rows=len(tr), pool_weight_ratio=1.0)`，
按实际样本平衡类别，并将 pool 总权重设为比赛训练折总权重。详见
[公开数据池用法](../../docs/open_pool.md)。
