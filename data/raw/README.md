# 原始数据（不提交到 Git）

从 Kaggle 比赛 **Data** 页下载后放到本目录：

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.parquet
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

数据文件较大且属于比赛数据，**不要提交到 Git**。
