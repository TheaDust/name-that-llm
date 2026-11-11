# A — Multi-view TF-IDF + LinearSVC

## 核心研究问题

如何用**词汇、字符和句式**差异识别 LLM？哪些特征块真正有区分力，哪些只是在拟合主题？

## 方法

1. **多视图特征**
   - `word`：TF-IDF 1–2 gram，`sublinear_tf=True`，去英文停用词；
   - `char`：`char_wb` 2–5 gram，捕捉标点、大小写、缩写和空格习惯；
   - `struct`：段长、句长分布、标点频率、换行/列表/标题模式、em dash、引号风格等。
2. **分类器**
   - LinearSVC 主模型（CPU 快、稀疏特征友好）；
   - 用 `CalibratedClassifierCV` 或 LogisticRegression 得到概率；
   - 需要的对比：LinearSVC vs LogisticRegression vs ComplementNB。
3. **消融**
   - word / char / struct / 两两组合 / 全部；
   - 特征维度、`min_df`、`max_df`、ngram 范围；
   - 每类召回率与混淆矩阵，重点看同厂商模型之间是否混淆。
4. **验证**
   - 严格按 `prompt_id` 分组；
   - 主指标使用完整 grouped OOF 的 balanced accuracy；
   - 输出完整 OOF 概率供融合。

## 资源

CPU，16 GB RAM；稀疏矩阵通常远小于内存上限。不需要 GPU。

## 产物

按根 README 的契约写入 `outputs/a_tfidf_svm/<run_id>/`，`run_id` 从 `a-tfidf-001` 开始。

## 目录

```text
experiments/a_tfidf_svm/
├── features.py  # 多视图 TF-IDF、结构特征
├── model.py     # LinearSVC / LogisticRegression / 概率校准
├── configs/     # 特征块、超参数、seed
├── notebooks/   # 个人探索
└── scripts/     # 训练/评估入口
```

## 待办

- [ ] 实现 `experiments/a_tfidf_svm/features.py`
- [ ] 实现 `experiments/a_tfidf_svm/model.py`
- [ ] 跑通 word-only baseline，登记第一个 OOF 分数
- [ ] 完成特征块消融与混淆矩阵
- [ ] 输出 `test_proba.parquet` 供融合
