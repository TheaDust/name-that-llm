# 仓库指南

## 项目结构与模块组织

- `src/ntl/` 存放共享的标签、数据校验、分组交叉验证、评估和提交工具。
- `experiments/a_tfidf_svm/`、`b_embedding_probe/` 和 `c_qlora/` 分别负责各路线专属的特征、模型、配置、Notebook 和训练脚本。只有当两条路线复用某段代码，或它定义了团队契约时，才把代码提升到 `ntl`。
- `scripts/` 存放共享的命令行编排；`tests/` 存放契约测试；`notebooks/` 存放共享分析。
- 原始输入放在 `data/raw/`，可复用缓存放在 `data/processed/`，运行产物放在 `outputs/<route>/<run_id>/`，最终提交放在 `submissions/`。模型存放约定见 `weights/README.md`。

## 开发命令

在仓库根目录使用 Python 3.10 或更高版本运行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[a,dev]"
python scripts/make_folds.py
python -m pytest tests/test_data.py -q
ruff check src scripts experiments tests
python scripts/make_submission.py --proba outputs/ensemble/test_proba.parquet
python scripts/validate_submission.py --submission submissions/submission.parquet
```

GPU 路线使用 `b` 或 `c` extras；请先安装与 CUDA 匹配的 PyTorch。fold 生成由一人统一完成，然后把文件分发给所有成员。

## 代码风格与命名约定

使用四空格缩进、`snake_case` 函数/文件名、`UPPER_CASE` 常量和描述性名称。Ruff 以 Python 3.10 为目标，行宽上限为 100 字符。优先采用最小完整实现；只有当前需要或已有复用基础时才增加抽象。CLI 脚本保持轻薄，路径保持可配置。

## 测试指南

使用 pytest，测试文件命名为 `test_*.py`。测试受影响的契约，尤其是 ID 对齐、prompt 隔离、标签和概率有效性。先运行受影响的模块；只有在核心改动、失败或已识别的回归风险时才扩大测试范围。仅修改文档时只需检查格式/链接，不需要运行测试。说明改了什么、为什么所选检查足够，以及为什么不需要更广泛的检查。

## 实验与协作约定

通过 `grouped_cv_splits(train, folds)` 读取共享 fold 表。在每个训练折内部拟合预处理和模型；评估完整的 OOF 预测。概率表的列为 `id` 加上 18 个固定的模型标签列。运行产物遵循 `README.md`，实验结果记录到 `docs/experiments.md`。数据、权重、缓存、日志和凭据不要提交到 Git。

## 提交与拉取请求指南

提交历史使用简短祈使句主题，并带 `feat:`、`refactor:`、`chore:` 等前缀。保持提交聚焦。PR 应说明改动、验证方式和相关实验结果；适用时关联相关 issue。共享代码的改动需要另一名团队成员评审。
