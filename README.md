# Name That LLM

Kaggle 竞赛 **Name That LLM** 的团队方案仓库：判断一段 40–250 词的英文文本由 18 个前沿大模型中的哪一个生成。

比赛规则、数据与评测细节见 [docs/competition.md](docs/competition.md)。

## 比赛要点

| 项目 | 说明 |
|---|---|
| 任务 | 18 类文本作者归因 |
| 训练集 | 2,699 条，150 个 `prompt_id`，同一 prompt 下每个模型一条 |
| 测试集 | 比赛单独提供；本地 3 行示例，正式约 4,000 条 |
| 指标 | balanced accuracy（18 类召回率平均） |
| 提交 | `submission.parquet`（`id`, `model`） |
| 限制 | Code Competition，CPU/GPU ≤ 9 小时，禁用互联网 |

## 团队分工

| 成员 | 模型路线 | 核心研究问题 | 资源 | 代码 |
|---|---|---|---|---|
| A | Multi-view TF-IDF + LinearSVC | 如何用词汇、字符和句式差异识别 LLM？ | CPU，16 GB RAM | `experiments/a_tfidf_svm/` |
| B | Qwen3-Embedding-8B + 多层特征分类器 | 预训练模型不同层的表示，哪些最能体现模型风格？ | 24 GB GPU 推荐 | `experiments/b_embedding_probe/` |
| C | Qwen3-4B/8B + QLoRA | 任务特定微调能否更好地区分 18 个模型？ | 16–24 GB GPU | `experiments/c_qlora/` |

路线专属代码放在各自的 `experiments/<route>/`。只有被至少两条路线复用，或属于全队共享契约的代码，才上移到 `src/ntl/`。

## 目录结构

```text
name-that-llm/
├── data/
│   ├── raw/                    # train/test/sample_submission（不提交）
│   └── processed/              # 共享 folds、特征缓存（不提交）
├── src/ntl/                    # 共享数据、指标、标签、提交契约
├── experiments/                # a_tfidf_svm / b_embedding_probe / c_qlora
├── scripts/                    # make_folds / make_submission / validate_submission
├── tests/                      # 数据、CV、概率、提交契约测试
├── docs/                       # competition / experiments / open_pool
├── weights/                    # 基座与最终权重（不提交）
├── notebooks/                  # 共享 EDA
├── submissions/                # 最终提交（不提交）
└── outputs/                    # 各 run 产物（不提交）
```

## 快速开始

```powershell
git clone https://github.com/TheaDust/name-that-llm.git
cd name-that-llm

python -m venv .venv
.venv\Scripts\Activate.ps1

# A 用 CPU；B/C 换成 .[b,dev] / .[c,dev]，并先装匹配 CUDA 的 PyTorch
pip install -e ".[a,dev]"

# 数据放到 data/raw/，详见 data/raw/README.md
python scripts/make_folds.py
python scripts/prepare_open_pool.py   # 可选：需要 data/raw/archive/
python scripts/check_data.py
python -m pytest -q
```

原始数据放到（下载步骤见 [data/raw/README.md](data/raw/README.md)）：

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.parquet
data/raw/archive/ntl_open_pool.parquet   # 可选
```

运行某条路线的训练入口：

```powershell
python -m experiments.a_tfidf_svm.scripts.train --config experiments/a_tfidf_svm/configs/baseline.yaml
```

## 共享代码

| 模块 | 内容 |
|---|---|
| `ntl.constants` | 18 个合法标签、列名、默认 fold 数与 seed |
| `ntl.data` | 数据加载/校验、共享 fold 读取、prompt-grouped CV |
| `ntl.open_pool` | 公开 open pool 清洗、类别权重、pool 内部 fold |
| `ntl.metrics` | balanced accuracy、每类召回、support 报告 |
| `ntl.submission` | 概率表校验、argmax、submission 校验/写出 |

```python
import pandas as pd

from ntl.data import grouped_cv_splits, load_train

train = load_train()
folds = pd.read_parquet("data/processed/folds.parquet")

for train_idx, val_idx in grouped_cv_splits(train, folds):
    tr, va = train.iloc[train_idx], train.iloc[val_idx]
    # 在 tr 内拟合预处理和模型，预测 va
```

## 验证与提交

- 验证使用固定 5 折，按 `prompt_id` 分组；读取 `data/processed/folds.parquet`，不能按行随机切分。
- 每个训练 id 必须恰好有一条 OOF 预测；主指标为 balanced accuracy，并附每类召回率。
- 概率表列 = `id` + 18 个模型名；每行和为 1，不允许 NaN。
- 每次 run 写入 `outputs/<route>/<run_id>/`：`config.yaml`、`folds.parquet`、`oof_proba.parquet`、`test_proba.parquet`、`metrics.json`、`log.txt`。
- 所有特征、标准化、PCA 和概率校准只能使用当前训练折。
- `test.csv` 不参与 CV；提交前用 `scripts/make_submission.py` 校验并写出 `submissions/submission.parquet`。

## 模型权重

| 内容 | 位置 |
|---|---|
| 基座模型 | Hugging Face 缓存（推荐）或 `weights/base/` |
| 单次 run 的 checkpoint / adapter | `outputs/<route>/<run_id>/` |
| 最终复用的 adapter | `weights/adapters/<run_id>/` |
| 最终合并模型 | `weights/merged/<run_id>/` |

建议将 `HF_HOME` 指向仓库外，例如 `D:\hf-cache`。权重全部不提交；Kaggle 提交时作为私有 Dataset / Model 挂载到 `/kaggle/input/...`。详见 [weights/README.md](weights/README.md)。

## 协作流程

1. 从 `main` 建分支：`feat/a-tfidf`、`feat/b-layer-probe`、`feat/c-qlora`。
2. 共享代码改动至少由另一名成员评审；路线代码留在各自 `experiments/`。
3. 实验结果记录到 [docs/experiments.md](docs/experiments.md)。
4. 不提交数据、权重、缓存、完整日志和凭据。

## 文档

- [比赛规则与数据](docs/competition.md)
- [公开 open pool 清洗与用法](docs/open_pool.md)
- [实验记录](docs/experiments.md)
- [路线 A](experiments/a_tfidf_svm/README.md) / [路线 B](experiments/b_embedding_probe/README.md) / [路线 C](experiments/c_qlora/README.md)
- [权重约定](weights/README.md)
