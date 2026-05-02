"""Beam-search expansion over next-character distributions."""

from __future__ import annotations

import heapq
import math
from pathlib import Path

import numpy as np

from passcrack.ml.data import Meta, encode_prefix

__all__ = ["beam_candidates", "load_model_bundle"]


def load_model_bundle(model_dir: Path | str):
    try:
        from tensorflow import keras  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("Install ML extra: pip install passcrack-edu[ml]") from e

    root = Path(model_dir)
    keras_path = root / "model.keras"
    meta_path = root / "meta.json"
    if not keras_path.is_file():
        raise FileNotFoundError(f"Missing {keras_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    model = keras.models.load_model(keras_path)
    meta = Meta.load(meta_path)
    return model, meta


def _predict_probs(model, prefix: str, meta: Meta) -> np.ndarray:
    enc = encode_prefix(prefix, meta["max_input_length"], meta["char_to_int"])
    x = np.array([enc], dtype=np.int32)
    # ``model(x)`` is far cheaper than ``predict`` for many small calls (beam search / evaluate).
    probs = model(x, training=False).numpy()[0].astype(np.float64)
    pad = int(meta["pad_idx"])
    probs[pad] = 0.0
    s = probs.sum()
    if s > 0:
        probs /= s
    return probs


def beam_candidates(
    model,
    meta: Meta,
    *,
    seed: str,
    beam_size: int,
    max_length: int,
    max_candidates: int,
) -> list[str]:
    """
    Beam-expand ``seed`` up to ``max_length`` characters.

    Returns up to ``max_candidates`` distinct strings ranked by best path negative log-probability.
    """
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if len(seed) > max_length:
        raise ValueError("max_length must be >= len(seed)")

    seen_best: dict[str, float] = {}
    frontier: list[tuple[float, str]] = [(0.0, seed)]

    steps = max(0, max_length - len(seed))
    for _ in range(steps):
        children: list[tuple[float, str]] = []
        for neg_logp, s in frontier:
            probs = _predict_probs(model, s, meta)
            idxs = np.argsort(-probs)[:beam_size]
            for idx in idxs:
                idx = int(idx)
                if idx == int(meta["pad_idx"]):
                    continue
                p = float(probs[idx])
                if p <= 1e-12:
                    continue
                ch = meta.int_to_char.get(idx)
                if ch is None:
                    continue
                ns = s + ch
                nlp = neg_logp - math.log(p + 1e-30)
                children.append((nlp, ns))
                prev = seen_best.get(ns)
                if prev is None or nlp < prev:
                    seen_best[ns] = nlp
        if not children:
            break
        frontier = heapq.nsmallest(beam_size, children)

    ranked = [s for s, _ in sorted(seen_best.items(), key=lambda kv: kv[1])]
    out: list[str] = []
    for s in ranked:
        if s not in out:
            out.append(s)
        if len(out) >= max_candidates:
            break
    return out
