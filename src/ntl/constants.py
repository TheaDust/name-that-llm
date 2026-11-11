"""Competition-wide constants shared by all three routes."""

MODEL_LABELS: tuple[str, ...] = (
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
)
N_MODELS = len(MODEL_LABELS)
MODEL_LABEL_SET = frozenset(MODEL_LABELS)

ID_COLUMN = "id"
TEXT_COLUMN = "text"
LABEL_COLUMN = "model"
PROMPT_COLUMN = "prompt_id"

PROBA_COLUMNS: tuple[str, ...] = (ID_COLUMN, *MODEL_LABELS)
DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 0
