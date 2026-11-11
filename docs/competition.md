# Name That LLM — Kaggle 比赛信息

> 来源：[https://www.kaggle.com/competitions/name-that-llm](https://www.kaggle.com/competitions/name-that-llm)
> 整理日期：2026-11-11

## 概述

18 个来自 6 家 AI 实验室的前沿大语言模型，能够生成从代码解释到创意小说的各类文本。随着模型生成内容充斥互联网和训练数据，判断一段文字由哪个模型撰写，对审计合成数据和检测基准污染至关重要。但仅凭一段短文区分前沿模型仍然非常困难。

传统 AI 检测器只区分“机器写”与“人写”。本任务更难：在**提示词未见、段落长度随机在 40–250 词之间**的情况下，指出确切的来源模型（authorship attribution）。不能依赖字符数或主题关键词等捷径；但每家实验室的训练数据和后训练选择，都会在词汇、措辞和标点上留下独特的风格指纹。

比赛要求对约 **4,000 条无标签文本段落**进行分类，候选为 **18 个前沿模型**，覆盖 OpenAI、Google、Anthropic、Qwen、ZAI 和 Moonshot AI。每条段落都是对哲学、物理、数学、生物学、计算机科学和创意写作等提示词的真实、自包含回答。

目标：展示仅凭短文本能多可靠地识别前沿模型；更好的归因方法将帮助研究者审计开放基准，并理解各模型家族的差异。

## 时间线

所有截止时间均为对应日期的 **UTC 23:59**（页面以新加坡时间显示）。

| 日期       | 事项                                    |
| ---------- | --------------------------------------- |
| 2026-10-14 | 比赛开始                                |
| 2027-01-06 | 参赛截止（须在此日期前接受比赛规则）    |
| 2027-01-06 | 组队/合并截止（最后一天加入或合并队伍） |
| 2027-01-13 | 最终提交截止                            |

主办方保留调整赛程的权利。

## 评估指标

提交结果按 **balanced accuracy（平衡准确率）** 评估：对每个候选模型，计算该类测试段落的召回率，再在 18 个模型上取平均。

$$
\text{Balanced Accuracy} = \frac{1}{K}\sum_{k=1}^{K}
\frac{\left|\{i : y_i = k \ \wedge\ \hat{y}_i = k\}\right|}
     {\left|\{i : y_i = k\}\right|}
$$

其中：

- $K = 18$：候选模型数量
- $y_i$：段落 $i$ 的真实作者模型
- $\hat{y}_i$：对段落 $i$ 的预测

注意：

- 测试集中**每个模型的段落数不相等**；
- 公共榜（public LB）和私有榜（private LB）的模型混合比例不同；
- balanced accuracy 让每个模型权重相同，避免多数类主导。

## 有效标签

预测值必须严格为以下 18 个模型名之一：

| 厂商        | 模型名                     |
| ----------- | -------------------------- |
| OpenAI      | `gpt-5.5`                |
| OpenAI      | `gpt-5.6-luna`           |
| OpenAI      | `gpt-5.6-sol`            |
| OpenAI      | `gpt-5.6-terra`          |
| OpenAI      | `gpt-6-astra`            |
| Google      | `gemini-3.1-pro-preview` |
| Google      | `gemini-3.6-flash`       |
| Google      | `gemini-3.7-flash`       |
| Google      | `gemini-3.8-flash`       |
| Anthropic   | `claude-fable-5`         |
| Anthropic   | `claude-opus-4-6`        |
| Anthropic   | `claude-opus-4-8`        |
| Anthropic   | `claude-opus-5`          |
| Anthropic   | `claude-sonnet-5`        |
| Qwen        | `qwen3.8-max`            |
| ZAI         | `glm-5.3`                |
| ZAI         | `glm-5.3-flash`          |
| Moonshot AI | `kimi-k3`                |

## 提交文件

这是 **Code Competition（代码竞赛）**：

- Notebook 从 `/kaggle/input/competitions/name-that-llm/test.csv` 读取测试集；
- 必须把 `submission.parquet` 写到 Notebook 的工作目录；
- 交互式运行时 `test.csv` 只含示例行；正式评分时会用完整隐藏测试集（约 4,000 条）重新运行；
- 每个 `id` 必须恰好出现一次，预测其作者 `model`。

格式（Parquet，两列）：

| id               | model                |
| ---------------- | -------------------- |
| `a1b2c3d4e5f6` | `gpt-6-astra`      |
| `b2c3d4e5f6a7` | `claude-opus-5`    |
| `c3d4e5f6a7b8` | `gemini-3.8-flash` |

最小示例：

```python
import pandas as pd

test = pd.read_csv('/kaggle/input/competitions/name-that-llm/test.csv')
submission = pd.DataFrame({'id': test['id'], 'model': predictions})
submission.to_parquet('submission.parquet', index=False)
```

出现以下情况会被拒绝：

- 缺少 `id` 或 `model` 列；
- 含空值；
- `id` 重复；
- 预测标签不在上述 18 个模型名之内。

## 代码要求

通过 Notebook 提交，提交前需满足：

- CPU Notebook 运行时间 ≤ 9 小时；
- GPU Notebook 运行时间 ≤ 9 小时；
- **禁用互联网访问**；
- 允许使用自由且公开可用的外部数据，包括预训练模型；
- 提交文件必须命名为 `submission.parquet`。

## 奖金

第 1–10 名各 **$5,000**，共 $50,000。

## 本地数据快照

当前目录中的文件与交互模式一致：

| 文件                          | 规模             | 列                                         |
| ----------------------------- | ---------------- | ------------------------------------------ |
| `train.csv`                 | 2,699 行         | `id`, `prompt_id`, `text`, `model` |
| `test.csv`                  | 3 行（交互示例） | `id`, `text`                           |
| `sample_submission.parquet` | 3 行             | `id`, `model`                          |

`train.csv` 的本地观察：

- 共 **150 个 `prompt_id`**（`p000`–`p149`）；
- 149 个 prompt 各有 18 条回答，`p002` 只有 17 条（缺 `claude-fable-5`）；
- 同一 prompt 内每个模型恰好一条，无重复；
- 每模型总行数为 150，`claude-fable-5` 为 149；
- `prompt_id` 只存在于训练集，测试集没有；它不是预测目标，只适合用于分组验证、组内对比学习或重建测试结构；
- 正式测试集约 4,000 条，且每模型条数不等，不能用训练集的“完整 18 条一组”直接假设测试集。

## 引用

Yao Yan, Martyna Plomecka, and Ashley Oldacre. *Name That LLM*. Kaggle, 2026.
[https://www.kaggle.com/competitions/name-that-llm](https://www.kaggle.com/competitions/name-that-llm)
