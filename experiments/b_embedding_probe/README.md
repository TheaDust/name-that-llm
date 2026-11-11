# B — Qwen3-Embedding-8B + 多层特征分类器

## 核心研究问题

预训练模型不同层的表示，**哪些层最能体现模型风格**？单层、层拼接、层选择分别能带来多少增益？

## 方法

1. **特征提取**
   - 加载 Qwen3-Embedding-8B，`output_hidden_states=True`；
   - 对每条文本取 last-token 与 mean pooling；
   - 先按层抽样（如每 4 层）跑通，再扩展到全部层；
   - 特征缓存到 `data/processed/qwen3_embedding_8b/`，文件名标注层号与 pooling。
2. **Probe**
   - 每层单独训练 LogisticRegression / LinearSVC；
   - 画 layer-wise balanced accuracy 曲线；
   - 尝试标准化、PCA、层拼接、top-k 层选择；
   - 与 A 的 TF-IDF 稀疏特征拼接，观察增益。
3. **验证**
   - 严格按 `prompt_id` 分组；
   - 提取特征时不使用标签，避免任何泄漏；
   - 主指标按隐藏测试形态（每 prompt 抽 1–3 条）计算。

## 资源

24 GB GPU 推荐。8B 模型 bf16 权重约 16 GB，加上激活和 batch 需要控制：

- 优先 batch=1–4、较短 `max_length`；
- 若 OOM，改用 4-bit 量化或只提取部分层；
- 推理/提取阶段关闭梯度。

## 产物

按根 README 的契约写入 `outputs/b_embedding_probe/<run_id>/`，`run_id` 从 `b-embed-001` 开始。Adapter/权重不进 Git。

## 目录

```text
experiments/b_embedding_probe/
├── embeddings.py # Qwen3-Embedding 各层 hidden states 提取
├── probe.py      # 逐层线性 probe / 层拼接 / 降维
├── configs/      # 模型、层列表、pooling、probe 超参
├── notebooks/    # 层级分析可视化
└── scripts/      # 提取、probe、评估入口
```

## 待办

- [ ] 实现 `experiments/b_embedding_probe/embeddings.py`
- [ ] 实现 `experiments/b_embedding_probe/probe.py`
- [ ] 跑通单层（如中间层）probe
- [ ] 完成 layer-wise 扫描与报告
- [ ] 与 A 的特征拼接对比
- [ ] 输出 `test_proba.parquet` 供融合
