# Name That LLM

Kaggle 竞赛 **Name That LLM** 的团队方案仓库：给定一段 40–250 词的英文文本，判断它由 18 个前沿大模型中的哪一个生成。

- 比赛规则、数据与评测整理：[docs/competition.md](docs/competition.md)
- 当前阶段：仓库脚手架 + 三条研究路线并行探索
- 评估指标：**balanced accuracy**（18 个模型召回率的平均）
- 最终目标：产出可复现的 OOF/测试概率，融合为 `submission.parquet`

## 比赛要点

| 项目     | 说明                                                          |
| -------- | ------------------------------------------------------------- |
| 任务     | 18 类文本作者归因（authorship attribution）                   |
| 候选模型 | OpenAI / Google / Anthropic / Qwen / ZAI / Moonshot 共 18 个  |
| 训练集   | 2,699 条，150 个 `prompt_id`，同一 prompt 下每个模型一条    |
| 隐藏测试 | 约 4,000 条，每个模型的条数不相等                             |
| 指标     | balanced accuracy，public/private 榜的模型混合比例不同        |
| 提交     | Code Competition，`submission.parquet`（`id`, `model`） |
| 限制     | CPU/GPU Notebook ≤ 9 小时，禁用互联网，允许公开外部数据      |

## 团队分工

| 成员 | 模型路线                            | 核心研究问题                                   | 主要资源       |
| ---- | ----------------------------------- | ---------------------------------------------- | -------------- |
| A    | Multi-view TF-IDF + LinearSVC       | 如何用词汇、字符和句式差异识别 LLM？           | CPU，16 GB RAM |
| B    | Qwen3-Embedding-8B + 多层特征分类器 | 预训练模型不同层的表示，哪些最能体现模型风格？ | 24 GB GPU 推荐 |
| C    | Qwen3-4B/8B + QLoRA                 | 通过任务特定微调，能否更好地区分 18 个模型？   | 16–24 GB GPU  |

### 代码归属

| 成员 | 路线代码 | 实验目录 | 实验产物 |
| ---- | -------- | -------- | -------- |
| A | `experiments/a_tfidf_svm/features.py`、`model.py` | `experiments/a_tfidf_svm/` | `outputs/a_tfidf_svm/` |
| B | `experiments/b_embedding_probe/embeddings.py`、`probe.py` | `experiments/b_embedding_probe/` | `outputs/b_embedding_probe/` |
| C | `experiments/c_qlora/qlora.py` | `experiments/c_qlora/` | `outputs/c_qlora/` |
| 全员 | `src/ntl/`（数据、指标、标签、提交契约）、`scripts/`、`tests/` | — | `outputs/ensemble/` |

原则：

- 每条路线在自己的 `experiments/<route>/` 下独立迭代，路线专属的特征、模型、训练逻辑也放在这里，避免互相踩文件；
- 只有**被至少两条路线复用**或属于**全队共享契约**的代码才放 `src/ntl/`；
- 单路线代码先留在原目录，等真正出现第二个使用方再上移，避免把“只有 A 用”的代码伪装成公共代码；
- 三路最终只通过统一的概率文件契约汇合，不在代码层面互相依赖。

## 研究路线

### A — Multi-view TF-IDF + LinearSVC

1. 多视图特征：word 1–2 gram、`char_wb` 2–5 gram、句长/段长/标点/换行等结构特征。
2. 用 LinearSVC 做主分类器，用 `CalibratedClassifierCV` 或 LogisticRegression 输出概率。
3. 做特征块消融（word / char / struct / 全部）、特征选择和混淆矩阵分析。
4. 目标：得到一个强、快、可解释的 CPU baseline，并作为 B/C 的对照。

### B — Qwen3-Embedding-8B + 多层特征分类器

1. 用 Qwen3-Embedding-8B 提取各层 hidden states，比较 last-token / mean pooling。
2. 对每一层单独训练线性 probe，画 layer-wise balanced accuracy，找出风格信息最强的层。
3. 尝试层拼接、PCA/标准化、与 A 的 TF-IDF 特征拼接。
4. 目标：回答“预训练表示的哪一层最像模型指纹”，并产出可与 A/C 融合的概率。

### C — Qwen3-4B/8B + QLoRA

1. 以 18 类分类为目标做 QLoRA 微调（分类头或标签 token logit scoring）。
2. 对比 zero-shot、线性 probe、QLoRA 三种设置。
3. 关注过拟合：必须使用 prompt-grouped CV，并做主题/长度切片分析。
4. 目标：验证任务特定微调是否真的带来稳定增益，产出 adapter 与概率文件。

## 统一实验契约

每个 run 的产物写到：

```text
outputs/<route>/<run_id>/
├── config.yaml          # 完整超参数、数据版本、随机种子
├── oof_proba.parquet    # 训练集 OOF 概率
├── test_proba.parquet   # 测试集概率
├── metrics.json         # 指标与评估协议
├── folds.parquet        # 本次使用的共享划分
└── log.txt              # 运行日志摘要
```

约定：

- `<route>` ∈ `a_tfidf_svm` / `b_embedding_probe` / `c_qlora`；`<run_id>` 形如 `a-tfidf-001`。
- `oof_proba.parquet`：列 = `id` + 18 个模型名，覆盖 `train.csv` 的每个 `id` 恰好一次，必须是**out-of-fold** 预测。
- `test_proba.parquet`：列 = `id` + 18 个模型名，覆盖 `test.csv` 的每个 `id` 恰好一次。
- 概率矩阵每行和为 1，不允许 NaN；列名必须与比赛标签完全一致。
- `metrics.json` 至少包含：`balanced_accuracy`、`per_class_recall`、`n_folds`、`seed`、`protocol`。
- `config.yaml` 记录超参数、训练 seed、数据文件 SHA-256、Git commit 和依赖版本；每个 run 同时保存实际使用的 `folds.parquet`。
- 特征拟合、标准化、PCA 和概率校准均只使用当前训练折；OOF 文件校验只检查格式，训练入口负责保证预测来源。
- 大文件（embedding 缓存、adapter、日志全文）不进 Git。

## 验证协议

同一个 prompt 的 18 条回答主题相同，只是模型不同，因此**绝不能按行随机划分**：

1. 用 `StratifiedGroupKFold` / `GroupKFold`，以 `prompt_id` 为 group 切分。
2. 每个验证折保留完整行用于生成 OOF 概率，汇总全部 OOF 后计算主指标。
3. 由一人生成并分发 `data/processed/folds.parquet`，各成员读取同一文件并按 `id` 对齐；每个 run 保存使用的 fold 文件。
4. `prompt_id` 只是训练集元数据，测试集没有，不能作为特征。
5. 报告 balanced accuracy 的同时，附上每类召回率和混淆矩阵。
6. 测试集每个模型的条数不相等，不能假设“一组 18 条、每模型一条”。

## 目录结构

```text
name-that-llm/
├── README.md
├── pyproject.toml              # 依赖与可选 extras（a / b / c / dev）
├── .gitignore
├── data/
│   ├── raw/                    # 原始 csv（不提交）
│   └── processed/              # 跨 run 复用的特征缓存（不提交）
├── weights/                    # 基座/最终权重（不提交，仅 README）
├── src/ntl/                    # 只放真正共享的代码
│   ├── __init__.py
│   ├── constants.py            # 18 个标签、列名
│   ├── data.py                 # 加载、校验、共享 prompt-grouped CV
│   ├── metrics.py              # balanced accuracy、每类召回、混淆矩阵
│   └── submission.py           # 概率校验、argmax、提交写出
├── experiments/                # 三条路线各自独立
│   ├── __init__.py
│   ├── a_tfidf_svm/            # 成员 A
│   │   ├── features.py         # 多视图 TF-IDF
│   │   ├── model.py            # LinearSVC / LogisticRegression
│   │   ├── configs/
│   │   ├── notebooks/
│   │   └── scripts/            # python -m experiments.a_tfidf_svm.scripts.train
│   ├── b_embedding_probe/      # 成员 B
│   │   ├── embeddings.py       # Qwen3-Embedding 特征提取
│   │   ├── probe.py            # 多层线性 probe
│   │   ├── configs/
│   │   ├── notebooks/
│   │   └── scripts/
│   └── c_qlora/                # 成员 C
│       ├── qlora.py            # QLoRA 微调与推理
│       ├── configs/
│       ├── notebooks/
│       └── scripts/
├── notebooks/                  # 共享 EDA，不放个人实验
├── scripts/                    # 共享入口：生成/校验 submission
├── tests/                      # 数据契约、指标、防泄漏测试
├── docs/
│   ├── competition.md          # 比赛规则与数据说明
│   └── experiments.md          # 实验记录表
├── submissions/                # 最终提交文件（不提交，仅保留 README）
└── outputs/                    # 所有 run 产物（整个目录不提交）
```

## 快速开始

```powershell
git clone <repo-url>
cd name-that-llm

python -m venv .venv
.venv\Scripts\Activate.ps1

# 按自己的路线安装；A 只用 CPU
pip install -e ".[a]"
# B:
# pip install -e ".[b]"
# C:
# pip install -e ".[c]"
```

说明：

- B/C 请先安装与本机 CUDA 匹配的 PyTorch，再安装其余依赖。
- `bitsandbytes` 在 Windows 上支持有限，C 建议在 Linux/WSL 或 Colab/Kaggle GPU 环境运行。
- 原始数据放到：

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.parquet
```

详细字段见 [data/raw/README.md](data/raw/README.md)。

各路线实现训练入口后，从仓库根目录以模块方式运行，例如：

```powershell
python -m experiments.a_tfidf_svm.scripts.train --config experiments/a_tfidf_svm/configs/baseline.yaml
```

## 共享前置代码（已实现）

| 模块 | 能力 |
|---|---|
| `ntl.constants` | 18 个合法标签、列名、默认 fold 数与 seed |
| `ntl.data` | 读取/校验 train/test、生成和使用共享 folds |
| `ntl.metrics` | 基于 sklearn 的 balanced accuracy、每类召回与 support 报告 |
| `ntl.submission` | 概率表校验、argmax、submission 校验/写出 |

典型用法：

```python
import pandas as pd

from ntl.data import load_train, grouped_cv_splits
from ntl.constants import MODEL_LABELS
from ntl.metrics import evaluate
from ntl.submission import write_proba

train = load_train()

folds = pd.read_parquet("data/processed/folds.parquet")

for train_idx, val_idx in grouped_cv_splits(train, folds):
    # 在 train.iloc[train_idx] 上拟合全部特征处理与模型，预测 train.iloc[val_idx]
    ...

# 将所有验证折预测按 train 的行顺序汇总成 oof（列为 id + 18 个标签）
# write_proba(oof, "outputs/a_tfidf_svm/a-tfidf-001/oof_proba.parquet", ids=train["id"])
# report = evaluate(train["model"], oof[list(MODEL_LABELS)].idxmax(axis=1))
```

生成全队共用的 fold 划分、跑测试、生成提交：

```powershell
python scripts/make_folds.py
python -m pytest -q

python scripts/validate_submission.py --submission submissions/submission.parquet
python scripts/make_submission.py `
    --proba outputs/ensemble/test_proba.parquet `
    --out submissions/submission.parquet
```

## 模型权重与缓存

| 内容 | 位置 | 是否提交 |
|---|---|---|
| 基座模型（Qwen3-Embedding-8B、Qwen3-4B/8B） | Hugging Face 缓存（推荐），或 `weights/base/` | 否 |
| 单次 run 的 checkpoint / adapter | `outputs/<route>/<run_id>/` | 否 |
| 最终选定、跨 run 复用的 adapter | `weights/adapters/<run_id>/` | 否 |
| 最终合并模型 | `weights/merged/<run_id>/` | 否 |
| 特征缓存 | `data/processed/` | 否 |

建议把 Hugging Face 缓存指到仓库外：

```powershell
$env:HF_HOME = "D:\hf-cache"
```

`weights/` 只有 README 会被 Git 跟踪；权重文件本身全部忽略。Kaggle 正式提交时需把基座模型和 adapter 作为私有 Dataset / Model 挂载到 `/kaggle/input/...`。详见 [weights/README.md](weights/README.md)。

## 协作流程

1. 从 `main` 建分支：`feat/a-tfidf`、`feat/b-layer-probe`、`feat/c-qlora`。
2. 路线专属代码放 `experiments/<route>/`；只有全队共享的契约代码放 `src/ntl/`。
3. 运行产物只写 `outputs/<route>/<run_id>/`，不要提交。
4. 在 [docs/experiments.md](docs/experiments.md) 登记 run 的指标和结论。
5. 发起 PR 合并回 `main`；共享代码至少一名其他成员 review。
6. 不提交数据、模型权重、embedding 缓存、完整日志和密钥。

## 结果汇总与提交

最终提交从各路线输出的 `test_proba.parquet` 生成：

1. 各路线概率先按 `id` 和固定标签列对齐，再在融合阶段实现平均或加权；
2. 用同一套完整 OOF 评估融合方案，再应用到测试概率；
3. 用 `scripts/make_submission.py` 校验并写出 `submissions/submission.parquet`；
4. 校验项：18 个合法标签、`id` 唯一、无空值、行数与 `test.csv` 一致。

## 近期里程碑

| 阶段 | 内容                                                        | 负责      |
| ---- | ----------------------------------------------------------- | --------- |
| 1    | 数据契约、指标、CV 切分、提交校验                           | 全员      |
| 2    | A 的 TF-IDF baseline；B 的 embedding 提取；C 的 QLoRA pilot | A / B / C |
| 3    | 各路特征与超参消融，统一 OOF 对比                           | A / B / C |
| 4    | 三路融合、错误分析、最终 Notebook                           | 全员      |
