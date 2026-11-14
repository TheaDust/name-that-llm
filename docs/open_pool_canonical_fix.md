# Open pool 跨来源分组修复

## 问题与修复

旧清洗流程直接使用来源内的 `prompt_group` 计算“至少 8 个模型”的覆盖度。
同一个 `prompt_canon` 在不同来源里的模型没有合并计数，导致当前原始 pool 中的
`gemini-3.7-flash` 样本全部被过滤掉。原始文件包含这一类，并非格式转换或标签映射错误。

修复后，在空文本、非法标签、长度、proxy、文本去重和冲突标签过滤之后：

1. 对非空 `prompt_canon` 做 NFKC 与空白归一化，保留大小写；相同值跨来源分组。
2. 使用带命名空间的 SHA-256 分组键，不做模糊匹配或语义匹配。
3. 缺失或空 canonical 元数据时，回退到原有来源内 `prompt_group`，不把空值合并。
4. 覆盖度过滤、每个 `(prompt, model)` 的样本上限和 pool 诊断分折使用同一套分组键。
5. CLI 在写入前检查全部 18 类；缺类默认报错。只有明确接受部分类别时，才使用
   `--allow-missing-classes`。报告记录缺失类别与分组策略版本。

输出列契约不变，但 `prompt_group` 的值及 pool 诊断分折会改变；`source_dataset`、
`source_license` 和原始行 ID 仍保留。原始来源内分组可通过 ID 回查 raw parquet。
比赛共享 folds 不受影响。没有 `prompt_canon` 列的旧输入继续使用原分组。

## 完整数据验证

2026-11-14，本机原始 pool 为 226,191 行。默认阈值下：

| 项目 | 旧流程 | 修复后 |
|---|---:|---:|
| 清洗后行数 | 64,952 | 82,318 |
| 覆盖类别 | 17 | 18 |
| `gemini-3.7-flash` 行数 | 0 | 1,203 |

新产物与比赛 train 无 ID 或归一化文本精确重叠；每个清洗后的 prompt 至少覆盖 8 类、
每个 prompt/model 最多 3 行，同组全部位于同一个 pool 诊断 fold。
这些检查不排除语义近重复或上游 canonical 元数据错误。

此修复对所有类别统一重新清洗，与此前“保留旧采样并单独补充目标类”的诊断实验不同。
不能把诊断实验的召回率直接当成新 pool 的训练成绩；使用新 pool 的模型仍需重新做比赛 OOF。

## 使用与迁移

先写入独立路径，保留原实验输入：

```bash
python scripts/prepare_open_pool.py \
  --out data/processed/open_pool_canonical_v1.parquet \
  --report data/processed/open_pool_canonical_v1.json
```

检查报告的 `classes`、`missing_classes`、`prompt_group_policy`、过滤步骤和输入/输出哈希。
后续训练需重新采样并记录新 pool 路径与哈希；原模型不会因脚本更新而自动改变。
默认执行入口同样采用新逻辑，运行前应备份需要保留的旧默认产物。

回归测试覆盖跨来源模型覆盖、空 canonical 回退、大小写区分、proxy 过滤后的覆盖度、
跨来源 pair cap、确定性、分组隔离，以及 CLI 缺类拒绝/显式放行。
运行 `python -m pytest tests/test_open_pool.py tests/test_training_data.py -q`。

这是共享代码修复，按仓库约定在合并前需要另一名团队成员评审。
