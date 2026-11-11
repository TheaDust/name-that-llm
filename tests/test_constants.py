from ntl.constants import MODEL_LABELS, MODEL_LABEL_SET, N_MODELS


def test_label_set_is_complete_and_unique():
    assert N_MODELS == 18
    assert len(MODEL_LABELS) == N_MODELS
    assert len(MODEL_LABEL_SET) == N_MODELS
    assert "claude-opus-5" in MODEL_LABEL_SET
    assert "kimi-k3" in MODEL_LABEL_SET
