# C — Qwen3 QLoRA 分类训练

使用 Qwen3 + 18 类分类头识别段落作者。模型直接读取段落文本，输出标签概率。
实现采用 4-bit NF4、双重量化和 LoRA，并训练、保存 `score` 分类头。

## 服务器环境

在仓库根目录操作，使用单张 NVIDIA CUDA GPU。先安装与服务器 CUDA 匹配的 PyTorch，
再安装本路线依赖：

```bash
pip install -e ".[c]"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data/hf-cache
```

框架使用 Transformers 4.x 的 Trainer 接口，依赖范围已写入 `pyproject.toml`。
本次只完成框架与静态检查，尚未在 GPU 上验证依赖组合、显存占用或训练效果。

## 准备数据

将比赛原始数据、共享 folds 和已经准备的 `training_splits/` 复制到服务器的相同相对目录。
目录中的 `manifest.json` 含原机器的来源路径，仅用于记录；训练读取 YAML 中的当前路径。
也可以在服务器准备数据：

```bash
python scripts/make_folds.py
python scripts/prepare_open_pool.py
python scripts/prepare_training_data.py
```

已有团队共享 folds 时直接复用。默认配置读取：

```text
data/raw/train.csv
data/raw/test.csv
data/processed/folds.parquet
data/processed/training_splits/manifest.json
data/processed/training_splits/fold_0/train.parquet
data/processed/training_splits/fold_0/validation.parquet
# 以及 fold_1 到 fold_4
```

训练入口会检查每折比赛训练/验证样本与共享 folds 的对应关系。
`sample_weight` 来自数据准备步骤，训练时只做一次全局归一化到均值 1，保持
pool 与比赛样本的相对贡献。loss 为逐样本交叉熵乘权重后求均值，使用普通随机采样。
验证 loss 和 balanced accuracy 使用原比赛验证样本。

## 配置

编辑 `configs/qwen3_4b.yaml`，所有相对路径相对于运行命令时的工作目录：

- `model.name_or_path`：默认 `Qwen/Qwen3-4B`，也可填写本地模型目录或换为 Qwen3-8B。
- `model.revision`：默认 `main`；固定版本实验时填写模型仓库 commit。
- `model.local_files_only`：服务器离线时设为 `true`，提前准备完整基座模型和 tokenizer。
- `model.compute_dtype`：默认 `bfloat16`；GPU 不支持时手动改为 `float16`。
- `model.max_length`：默认 1024，长文本截断；显存不足时可降低长度或 batch。
- `data.splits_dir`：指向想使用的数据准备目录，包括全量 pool 或仅比赛数据的对照。
- `training`：默认三轮、学习率 1e-4、batch=1、梯度累积 16，按 epoch 验证与保存。
- `training.early_stopping_patience`：默认 1，按验证 balanced accuracy 早停并加载最佳权重。
- `output_dir`：每次实验使用一个新目录，已有目录不会被覆盖。

数据采样量和 pool 权重比例在数据准备阶段决定，训练脚本使用 parquet 中的全部训练行。
仅 `text` 进入 tokenizer；ID、来源、prompt 分组均作为元数据。

## 运行

先在服务器单折检查显存、速度和数据接入：

```bash
python -m experiments.c_qlora.scripts.train \
  --config experiments/c_qlora/configs/qwen3_4b.yaml \
  --fold 0 --output-dir outputs/c_qlora/c-qlora-pilot
```

完整五折：

```bash
python -m experiments.c_qlora.scripts.train \
  --config experiments/c_qlora/configs/qwen3_4b.yaml
```

各折独立重新加载基座和初始化 adapter，按相同标签顺序输出 softmax 概率。
默认使用固定 seed。完整运行后，以各折最佳模型的测试概率平均作为测试预测。
单折模式只生成该折产物，不能用来报告完整 OOF 分数。

## 产物

```text
outputs/c_qlora/c-qlora-001/
├── config.yaml              # 实际模型、数据、训练配置与所选折
├── data_manifest.json       # 数据准备配置
├── inputs.json              # 当前输入文件 SHA-256
├── environment.json         # GPU、CUDA 与依赖版本
├── folds.parquet            # 实际共享 folds 的副本
├── log.txt
├── fold_0/
│   ├── checkpoints/         # Trainer checkpoint
│   ├── adapter/             # LoRA、分类头、模型配置、tokenizer
│   ├── history.json         # 训练/验证日志记录
│   ├── metrics.json
│   ├── oof_proba.parquet    # 该折验证预测
│   └── test_proba.parquet
├── fold_1/ ... fold_4/
├── metrics.json             # 完整 OOF balanced accuracy、逐类召回、各折结果
├── oof_proba.parquet        # 按原比赛 train ID 顺序对齐
└── test_proba.parquet       # 五折测试概率平均
```

根目录的三项汇总产物只在完整五折模式生成。概率表沿用共享契约：`id` + 18 个标签列。
adapter 需要配合原始 Qwen3 基座加载，不能作为完整基座模型使用。

在服务器训练完成后生成提交：

```bash
python scripts/make_submission.py \
  --proba outputs/c_qlora/c-qlora-001/test_proba.parquet \
  --out submissions/c-qlora-001.parquet
```

训练结果确认后记录到 `docs/experiments.md`。数据、adapter、checkpoint 和完整日志不提交 Git。

## 实现参考

- [Transformers Qwen3](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen3)
- [PEFT 量化训练](https://huggingface.co/docs/peft/en/developer_guides/quantization)
