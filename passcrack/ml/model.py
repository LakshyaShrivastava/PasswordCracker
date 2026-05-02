"""Small many-to-one LSTM next-character model."""

from __future__ import annotations

__all__ = ["build_model"]


def build_model(max_input_length: int, vocab_size: int):
    try:
        from tensorflow import keras  # type: ignore[import-untyped]
        from tensorflow.keras import Sequential  # type: ignore[import-untyped]
        from tensorflow.keras.layers import Dense, Embedding, LSTM  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("Install ML extra: pip install passcrack-edu[ml]") from e

    model = Sequential(
        [
            Embedding(vocab_size, 64),
            LSTM(128),
            Dense(vocab_size, activation="softmax"),
        ],
        name="next_char_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
