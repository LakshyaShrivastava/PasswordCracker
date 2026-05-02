import pytest

pytest.importorskip("tensorflow", reason="TensorFlow optional ML extra")

import numpy as np

from passcrack.ml.model import build_model


def test_next_char_model_forward():
    model = build_model(max_input_length=8, vocab_size=32)
    x = np.zeros((2, 8), dtype=np.int32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 32)
