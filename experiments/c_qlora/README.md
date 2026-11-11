# C — Qwen3-4B/8B + QLoRA

## 核心研究问题

通过任务特定的 QLoRA 微调，能否比零样本/线性 probe 更好地区分 18 个模型？增益来自风格信号，还是仅仅记住了主题？

## 方法

1. **任务形式**
   - 优先：`AutoModelForSequenceClassification` + 18 类分类头；
   - 备选：把 18 个模型名作为标签 token，用 logit scoring；
   - 避免自由生成模型名，生成结果不稳定且难以校准。
2. **QLoRA 配置（起点）**
   - 4-bit NF4 量化，双量化；
   - LoRA `r=16/32`、`alpha=32`、`dropout=0.05`；
   - target modules 覆盖 attention 的 q/k/v/o 与 MLP；
   - `lr=1e-4`、1–3 epochs、`max_length=1024`、bf16。
3. **训练策略**
   - 类别平衡采样或 weighted loss；
   - 早停依据 grouped validation 的 balanced accuracy；
   - 对比：zero-shot、线性 probe、QLoRA 全量/分层学习率。
4. **验证与解释**
   - 严格按 `prompt_id` 分组；
   - 主指标使用完整 grouped OOF 的 balanced accuracy；
   - 做长度、主题、模型家族的切片分析，确认不是主题捷径。

## 资源

16–24 GB GPU：

- Qwen3-4B + QLoRA 在 16 GB 上更稳；
- Qwen3-8B + QLoRA 在 24 GB 上需要控制 batch 和序列长度；
- 建议在 Linux/WSL 或 Colab/Kaggle GPU 上运行（`bitsandbytes` 的 Windows 支持有限）。

## 产物

按根 README 的契约写入 `outputs/c_qlora/<run_id>/`，`run_id` 从 `c-qlora-001` 开始。单次 run 的 adapter 留在 `outputs/`；最终选定、需要复用的 adapter 复制到 `weights/adapters/<run_id>/`。权重与缓存不进 Git。

## 目录

```text
experiments/c_qlora/
├── qlora.py     # 模型加载、LoRA 注入、训练与推理
├── configs/     # 模型、LoRA、训练与推理超参
├── notebooks/   # 训练曲线与错误分析
└── scripts/     # 训练、推理、评估入口
```

## 待办

- [ ] 确定分类头方案并跑通小样本
- [ ] 实现 `experiments/c_qlora/qlora.py`
- [ ] Qwen3-4B QLoRA pilot，登记 OOF
- [ ] Qwen3-8B 对比与显存调优
- [ ] 与 A/B 的 OOF 结果对比
- [ ] 输出 `test_proba.parquet` 供融合
