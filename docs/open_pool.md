# 公开数据池 open pool

外部数据来源：[`crtproduction/name-that-llm-open-pool`](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool)

它把 61 个公开的 Name That LLM 相关数据集合并成一份 18 类语料。**它不是比赛 train/test**，是允许使用的公开外部数据；仓库只保存清洗脚本，不保存原始和清洗后的 parquet。

> 当前默认清洗逻辑已修复跨来源 prompt 覆盖度计算，并默认拒绝缺类产物。以下行数及 17 类描述为旧流程记录；新规则、验证结果和迁移方式见 [跨来源分组修复](open_pool_canonical_fix.md)。

## 历史探索结果

以下是旧版数据与权重方案的探索记录，运行配置和 OOF 尚未归档，不能作为当前方案的效果结论。当前方案需要重新运行比赛 OOF 对照。

在比赛 `train.csv` 上做 5 折 prompt-grouped CV，TF-IDF + LinearSVC，每类最多取 1,200 行：

| 训练数据 | balanced accuracy |
|---|---|
| 仅 `train.csv` | 0.2674 |
| `train.csv` + 清洗后 pool | **0.3238** |

逐类召回变化（节选）：

| 模型 | 仅 train | + pool | Δ |
|---|---|---|---|
| qwen3.8-max | 0.200 | 0.493 | +0.293 |
| gemini-3.8-flash | 0.287 | 0.533 | +0.247 |
| claude-opus-4-6 | 0.233 | 0.453 | +0.220 |
| gemini-3.6-flash | 0.407 | 0.533 | +0.127 |
| claude-opus-4-8 | 0.120 | 0.233 | +0.113 |
| gpt-6-astra | 0.393 | 0.507 | +0.113 |
| kimi-k3 | 0.247 | 0.147 | -0.100 |
| gemini-3.7-flash | 0.340 | 0.113 | -0.227 |

效果评估同时报告整体 balanced accuracy 和逐类召回。

## 清洗规则

入口：

```powershell
python scripts/prepare_open_pool.py
```

默认过滤：

| 步骤 | 规则 | 删除行数 |
|---|---|---|
| 原始 | `ntl_open_pool.parquet` | 223,840 |
| 空文本 / 非法标签 | 必须有 text 和 18 类之一 | 0 |
| 长度 | 40–250 词 | -3,583 |
| proxy 生成 | 去掉 `flag_kb_proxy` 或 `flag_kb_proxy_after_ruling` | -83,012 |
| 同文本多标签 | 标签冲突的行全部丢弃 | 0 |
| 重复文本 | NFKC + 空白归一化后去重 | 0 |
| prompt 太薄 | prompt 内至少 8 个不同模型 | -71,629 |
| 单对上限 | 每个 `(prompt, model)` 最多 3 行 | -664 |
| 最终 | | **64,952** |

补充说明：

- `gemini37_unverified.parquet` 有 10,838 行，作者已标为路由未验证，**禁止使用**。
- 只有 1 个模型的 prompt 一律不用：prompt 本身就是标签捷径。
- 当前产物覆盖 17 类；`gemini-3.7-flash` 的训练样本来自比赛训练折。模型输出仍为固定 18 类。
- 已校验与 `train.csv` 无 id 重叠、无文本重叠。
- 近重复、跨来源重复仍可能存在，当前只做精确去重。

## 产物

```text
data/processed/open_pool_clean.parquet   # 清洗结果（不提交 Git）
data/processed/open_pool_clean.json      # 配置、每步行数、SHA-256
```

字段：

| 列 | 含义 |
|---|---|
| `id` | pool 内唯一 id，与 train id 不重叠 |
| `text` | 归一化后的 40–250 词英文段落 |
| `model` | 18 类之一 |
| `prompt_group` | pool 内的题目分组 |
| `n_words` | 词数 |
| `source_dataset` | 原始 Kaggle 数据集 |
| `source_license` | CC BY 4.0 / CC BY-SA 4.0 / CC0 |
| `style_score` | 数据集作者给出的风格偏离分，通常越小越接近比赛文本 |
| `match_weight` | 数据集作者给出的匹配分，用于清洗时的回答排序 |
| `pool_fold` | 仅用于 pool 内部诊断的 5 折 |

## 三条路线怎么用

共同规则：

1. pool 只加入**训练折**，所有最终指标在比赛 `train.csv` 的共享 5 折 OOF 上计算。
2. 先每类最多取 800–1,200 行做对照，再考虑全量；保存实际采样 ID 和 seed。
3. 每次采样或过滤后，按实际选中的样本调用 `pool_sample_weights()`。
4. 比赛训练行权重为 1；默认 pool 总权重与当前比赛训练折总权重相等。
   `pool_weight_ratio=0.5` 表示 pool 总权重为比赛训练折的一半。
   这个比例是实验起点，效果需要 OOF 验证。
5. pool 内每个有样本的类别获得相同总权重；采样已平衡时，每行权重相同。
   使用这些权重时，分类器保持 `class_weight=None`。
6. 清洗产物保存数据与来源，训练权重在运行时计算；`match_weight` 只用于清洗时
   选择同题同模型的回答，不直接作为训练权重。

### 统一准备入口

先完成清洗与共享 fold 生成，再运行：

```powershell
python scripts/prepare_training_data.py
```

默认每类最多取 1,000 行，五折共用同一份采样。产物位于
`data/processed/training_splits/`，每折包含 `train.parquet` 和 `validation.parquet`。
训练表已包含 `sample_weight`；`manifest.json` 记录输入 SHA-256、配置和各折统计，
`sampled_pool.parquet` 保存实际选择的样本及来源。输出目录必须尚不存在。
完整参数和读取示例见 [共享脚本](../scripts/README.md)。

### 路线 A：TF-IDF + LinearSVC

可直接读取导出的训练/验证表，或在 Python 中调用同一接口：

```python
import pandas as pd

from ntl.data import load_train
from ntl.open_pool import load_clean_pool
from ntl.training_data import sample_training_pool, training_folds

train = load_train()
folds = pd.read_parquet("data/processed/folds.parquet")
pool = sample_training_pool(
    load_clean_pool("data/processed/open_pool_clean.parquet"),
    max_rows_per_class=1000,
    seed=0,
)
for fold, training, validation in training_folds(train, folds, pool, pool_weight_ratio=1.0):
    # vectorizer 只 fit 在 training["text"] 上
    # model.fit(X_train, training["model"], sample_weight=training["sample_weight"])
    # 验证只用 validation
    ...
```

随机按类采样后，部分 prompt 的模型覆盖可能低于清洗阈值；清洗阈值描述的是完整产物。
保存采样后的 prompt 覆盖统计，避免把它当作仍满足每题至少 8 类的保证。

### 路线 B：Qwen3-Embedding + 多层特征

沿用 A 的样本选择和权重计算。embedding 按样本 ID 缓存到
`data/processed/qwen3_embedding_8b/pool/`，与训练特征按 ID 对齐后拼接。
每折向 probe 传入比赛行权重 1 和当折计算的 pool 权重。
标准化、PCA 等预处理只在当前训练折与所选 pool 上拟合。

### 路线 C：Qwen3 QLoRA

分类微调可从每类最多 500 行开始，当前覆盖 17 类，共约 8,500 条外部数据。
训练入口直接使用已准备的权重，全局归一化到均值 1 后，
对逐样本交叉熵乘权重求均值，保留 pool 与比赛数据的权重比例。
训练样本均匀抽取，避免再叠加类别平衡采样。运行方式见
[QLoRA 训练说明](../experiments/c_qlora/README.md)。

pool 总权重约束针对监督分类训练。单独使用 pool 文本做风格适配属于另一项实验，
应单独记录配置，并以比赛 OOF 判断效果。

`pool_fold` 只用于池内诊断；模型选择和早停依据比赛训练数据的验证划分。

## 已知风险

- **标签噪声**：pool 的标签来自 61 个来源，作者自己的测试也只有约 0.47 balanced accuracy。它是辅助数据，不是金标准。
- **prompt  shortcut**：即使过滤到每个 prompt ≥8 个模型，prompt 本身仍可能携带主题信息；不要用 pool 的随机行做验证。
- **来源集中**：清洗后来源仍较集中。部分模型高度依赖单一来源，例如 `claude-opus-5` 约 64%，可能学到来源指纹。来源上限或来源降权应作为单独的 OOF 对照实验。
- **风格偏移**：pool 的标点、数字、括号比例与比赛文本仍有差距，`style_score` 可用来做进一步过滤或降权。
- **许可**：约 63% 是 CC BY 4.0，约 32% 是 CC BY-SA 4.0；当前清洗结果里没有 CC0 行。仓库不提交数据；如需在 Notebook 外分发清洗产物，保留来源和许可信息。
- **隐藏测试分布未知**：pool 的 prompt 与 train 不同，不能保证与隐藏测试同分布。

## 复现

```powershell
# 原始文件放到 data/raw/archive/ntl_open_pool.parquet
python scripts/prepare_open_pool.py
python scripts/check_data.py

# 查看报告
Get-Content data/processed/open_pool_clean.json
```

常用参数：

```powershell
python scripts/prepare_open_pool.py --max-rows-per-pair 2
python scripts/prepare_open_pool.py --min-prompt-models 10
```
