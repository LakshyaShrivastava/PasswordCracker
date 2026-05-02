"""Train next-character LSTM and write ``model.keras`` + ``meta.json``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from passcrack.ml.data import Meta, build_char_maps, load_passwords_from_file, make_training_pairs
from passcrack.ml.model import build_model

__all__ = ["train_and_save"]


def train_and_save(
    wordlist: Path | str,
    out_dir: Path | str,
    *,
    epochs: int = 40,
    max_input_length: int = 16,
    validation_split: float = 0.1,
    batch_size: int = 64,
) -> None:
    passwords = load_passwords_from_file(wordlist)
    if len(passwords) < 2:
        raise ValueError("Need at least two passwords in wordlist for training")

    char_to_int = build_char_maps(passwords)
    xs, ys = make_training_pairs(passwords, max_input_length, char_to_int)
    X = np.array(xs, dtype=np.int32)
    y = np.array(ys, dtype=np.int32)

    vocab_size = len(char_to_int) + 1
    model = build_model(max_input_length, vocab_size)
    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    model.save(root / "model.keras")
    Meta(max_input_length=max_input_length, char_to_int=char_to_int).save(root / "meta.json")
