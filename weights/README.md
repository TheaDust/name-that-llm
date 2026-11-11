# 模型权重与缓存（不提交到 Git）

本目录用于存放**需要在多个 run 之间复用**的模型权重。只有本 README 会被 Git 跟踪，其余文件全部忽略。

## 建议布局

```text
weights/
├── README.md
├── base/                    # 本地基座模型快照（可选）
│   ├── Qwen3-Embedding-8B/
│   └── Qwen3-4B/
├── adapters/                # 最终选定、需要复用的 LoRA adapter
│   └── c-qlora-001/
└── merged/                  # 最终合并后的完整模型（可选）
    └── c-qlora-001/
```

## 放置规则

| 内容 | 位置 | 是否提交 |
|---|---|---|
| 基座模型 | Hugging Face 缓存（推荐），或 `weights/base/` | 否 |
| 单次 run 的 checkpoint / adapter | `outputs/<route>/<run_id>/` | 否 |
| 最终选定、跨 run 复用的 adapter | `weights/adapters/<run_id>/` | 否 |
| 最终合并模型 | `weights/merged/<run_id>/` | 否 |
| 特征缓存 | `data/processed/` | 否 |

## Hugging Face 缓存

建议把缓存指到仓库外，避免误提交和仓库膨胀：

```powershell
$env:HF_HOME = "D:\hf-cache"
```

Linux / WSL：

```bash
export HF_HOME=/data/hf-cache
```

## Kaggle 提交

正式提交时 Notebook 没有互联网，基座模型和 adapter 需要作为私有 Kaggle Dataset / Model 上传，并在 Notebook 中从 `/kaggle/input/...` 加载。不要把权重直接放进 Git 仓库。
