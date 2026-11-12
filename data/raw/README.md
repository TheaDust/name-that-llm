# 原始数据（不提交到 Git）

## 目录结构

```text
name-that-llm/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.parquet
│   │   └── archive/                       # 可选，公开 open pool
│   │       ├── ntl_open_pool.parquet      # 清洗脚本唯一需要的文件
│   │       ├── sources.csv
│   │       ├── source_protocol.csv
│   │       ├── removal_summary.csv
│   │       └── per_model.csv
│   └── processed/                         # 脚本生成，不提交
│       ├── folds.parquet
│       ├── open_pool_clean.parquet
│       └── open_pool_clean.json
├── outputs/                               # 各 run 产物
└── submissions/                           # 最终提交
```

## 0. 安装环境

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[a,dev]"
```

B/C 路线把 `a` 换成 `b` 或 `c`，并先安装匹配 CUDA 的 PyTorch。

## 1. 下载比赛数据

先在比赛页面接受规则，否则会 403。

网页：比赛 **Data** 页 → Download All → 解压到 `data/raw/`。

Kaggle CLI：

```powershell
pip install kaggle

# 凭证放到 %USERPROFILE%\.kaggle\kaggle.json
kaggle competitions download -c name-that-llm -p data/raw
Expand-Archive data/raw/name-that-llm.zip -DestinationPath data/raw -Force
```

Linux / macOS 用 `unzip data/raw/name-that-llm.zip -d data/raw`。

放好后：

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.parquet
```

如果比赛给的是 `sample_submission.csv`，转换：

```powershell
python -c "import pandas as pd; pd.read_csv('data/raw/sample_submission.csv').to_parquet('data/raw/sample_submission.parquet', index=False)"
```

## 2. 下载 open pool（可选）

公开数据集，不需要接受比赛规则：

```powershell
kaggle datasets download -d crtproduction/name-that-llm-open-pool `
    -p data/raw/archive --unzip
```

必须保证这个路径存在：

```text
data/raw/archive/ntl_open_pool.parquet
```

如果解压后多了一层 `archive/`，把 `ntl_open_pool.parquet` 移到 `data/raw/archive/` 下即可。

清洗：

```powershell
python scripts/prepare_open_pool.py
```

只使用 `ntl_open_pool.parquet`。`gemini37_unverified.parquet` 已被作者标记为路由未验证，**不要使用**。详见 [docs/open_pool.md](../../docs/open_pool.md)。

## 3. 生成共享 fold

```powershell
python scripts/make_folds.py
```

输出：

```text
data/processed/folds.parquet
```

5 折按 `prompt_id` 分组，预期每折：

```text
fold 0: 540 行 / 30 prompts
fold 1: 540 行 / 30 prompts
fold 2: 540 行 / 30 prompts
fold 3: 539 行 / 30 prompts
fold 4: 540 行 / 30 prompts
```

### 训练集 / 验证集怎么分

`folds.parquet` 固定了每一行的 fold（0–4），规则是：

- 按 `prompt_id` 分组：同一个 prompt 的 17–18 条回答必须在同一折；
- 分层：尽量让每折的 18 类比例接近；
- 固定 5 折、`seed=0`，由 `ntl.data.assign_folds()` 用 `StratifiedGroupKFold` 生成。

训练时轮流拿一折做验证：

```text
第 1 轮：fold 0 验证，fold 1+2+3+4 训练
第 2 轮：fold 1 验证，fold 0+2+3+4 训练
第 3 轮：fold 2 验证，fold 0+1+3+4 训练
第 4 轮：fold 3 验证，fold 0+1+2+4 训练
第 5 轮：fold 4 验证，fold 0+1+2+3 训练
```

每次验证集约 540 行 / 30 prompts，训练集约 2,159 行 / 120 prompts。

如果使用 open pool：

```text
训练集 = 4 折 + pool
验证集 = 1 折（pool 永远不进入验证）
```

5 轮跑完后，2,699 行每行恰好被验证一次，拼成完整 OOF。`test.csv` 不参与这个划分，它是比赛单独提供的测试集。

不要按行随机切分：同一个 prompt 的兄弟回答同时出现在训练和验证，会让分数虚高。

这个文件应由一人生成后分发给全队；其他人也可以重跑，但必须先确认 `train.csv` 是同一份。

## 4. 验证目录

```powershell
python scripts/check_data.py
```

全部就绪时输出：

```text
OK       train.csv                      2699 rows  ...
OK       test.csv                          3 rows  ...
OK       sample_submission.parquet         3 rows  ...
OK       folds.parquet                  2699 rows  ...
OK       open_pool_clean.parquet        64952 rows  ...
OK: data layout is ready
```

没有 open pool 时：

```powershell
python scripts/check_data.py --skip-pool
```

数据放在仓库外：

```powershell
$env:NTL_DATA_DIR = "D:\ntl-data\raw"
python scripts/check_data.py
```

## 字段

| 文件 | 列 | 说明 |
|---|---|---|
| `train.csv` | `id` | 段落唯一 id |
|  | `prompt_id` | 同题分组标记（仅训练集有） |
|  | `text` | 模型生成的段落 |
|  | `model` | 真实作者模型，18 类之一 |
| `test.csv` | `id` | 段落唯一 id |
|  | `text` | 待预测段落 |
| `sample_submission.parquet` | `id` | 测试段落 id |
|  | `model` | 预测标签占位 |

## 本地快照

- `train.csv`：2,699 行，150 个 `prompt_id`；149 个 prompt 各 18 条，`p002` 为 17 条（缺 `claude-fable-5`）。
- 同一 prompt 内每个模型恰好一条，适合做组内对比/平衡分配。
- `test.csv`：本地交互模式只有 3 行示例；正式评分约为 4,000 条。
- 正式测试集每个模型的条数不相等，public/private 榜的模型比例也不同。
- `prompt_id` 不是预测目标，测试集也没有这一列，不能作为模型特征。

## archive 里哪些文件有用

| 文件 | 用途 | 是否必需 |
|---|---|---|
| `ntl_open_pool.parquet` | 清洗脚本的唯一输入 | **是** |
| `sources.csv` | 61 个来源与许可，写说明/引用时用 | 建议保留 |
| `source_protocol.csv` | 各来源的生成协议 | 建议保留 |
| `removal_summary.csv` | 作者删了哪些行、为什么 | 建议保留 |
| `per_model.csv` | 每类行数、来源数、proxy 数 | 可选 |
| `prompts.parquet` | prompt 文本与 panel 信息，做 EDA 用 | 可选 |
| `dataq_results.csv` / `dataq4_paired.csv` | 作者的质量实验 | 可选 |
| `style_gap.csv` | 与比赛文本的风格差距 | 可选 |
| `gemini37_unverified.parquet` | 路由未验证，**禁止使用** | 否 |
| `removed_rows.csv` | 被删行的审计记录 | 否 |
| `fig*.png`、`v1*_append_log.csv` | 图表和版本日志 | 否 |

其他人如果只拿 `data/processed/open_pool_clean.parquet`，整个 `archive/` 都可以没有；只有需要重新清洗的人才需要保留 `ntl_open_pool.parquet`。

数据文件较大且属于比赛数据，**不要提交到 Git**。
