# 共享脚本

已提供以下入口：

| 脚本 | 作用 |
|---|---|
| `make_folds.py` | 生成统一的 prompt-grouped CV 划分，缓存到 `data/processed/folds.parquet` |
| `prepare_open_pool.py` | 清洗公开 open pool，写出 `data/processed/open_pool_clean.parquet` 与报告 |
| `prepare_training_data.py` | 固定采样 pool，拼接各折训练数据并计算权重，导出训练/验证 parquet 和配置记录 |
| `check_data.py` | 检查数据目录、schema、fold 对齐和 open pool 是否就绪 |
| `make_submission.py` | 读取 `test_proba.parquet`，argmax 后校验并写出 `submissions/submission.parquet` |
| `validate_submission.py` | 校验列名、id 覆盖、空值、标签合法性 |

用法：

```powershell
python scripts/make_folds.py
python scripts/prepare_open_pool.py
python scripts/check_data.py
python scripts/prepare_training_data.py
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

## 准备训练与验证数据

```powershell
python scripts/prepare_training_data.py
# 仅比赛数据的对照
python scripts/prepare_training_data.py --competition-only --out data/processed/competition_splits
# 调整采样量与 pool 总权重比例
python scripts/prepare_training_data.py --max-pool-rows-per-class 500 --pool-weight-ratio 0.5 --out data/processed/pool500_splits
```

默认读取比赛 `train.csv`、共享 `folds.parquet` 和清洗后的 pool，每类最多采样 1,000 行，
seed=0，五折共用同一批 pool。默认 pool 总权重与当折比赛训练部分相等。
输出目录必须尚不存在，避免不同配置混用。`--train`、`--folds`、`--pool`、`--out` 可指定路径。

```text
data/processed/training_splits/
├── manifest.json            # 输入路径与 SHA-256、采样参数、各折行数与权重总量
├── sampled_pool.parquet     # 固定采样结果及来源信息；仅比赛数据模式不生成此文件
├── fold_0/
│   ├── train.parquet
│   └── validation.parquet
└── fold_1/ ... fold_4/
```

`train.parquet` 的 `text` 是输入、`model` 是标签，训练时传入 `sample_weight`。
`data_source` 区分 `competition` / `pool`；`prompt_id` 和 `prompt_group` 分别保存两类数据的分组，
这些元数据不作为模型特征。`validation.parquet` 保持原比赛字段，只包含当前验证折。

```python
import pandas as pd

training = pd.read_parquet("data/processed/training_splits/fold_0/train.parquet")
validation = pd.read_parquet("data/processed/training_splits/fold_0/validation.parquet")
# 在 training["text"] 上拟合特征，使用 training["model"] 和 training["sample_weight"] 训练。
# 在 validation["text"] 上预测，并与 validation["model"] 比较。
```
