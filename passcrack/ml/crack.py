"""Try ML-ranked candidates against a target digest."""

from __future__ import annotations

import heapq
import math
from pathlib import Path

import numpy as np

from passcrack.hashing import matches
from passcrack.ml.data import Meta
from passcrack.ml.generate import _predict_probs, load_model_bundle

__all__ = ["crack_with_ml", "crack_with_ml_bundle"]


def crack_with_ml_bundle(
    model,
    meta: Meta,
    target: str,
    algo: str,
    *,
    seed: str,
    beam_size: int,
    max_length: int,
    budget: int,
    salt: str | None = None,
) -> str | None:
    trials = 0

    def try_plaintext(guess: str) -> str | None:
        nonlocal trials
        if trials >= budget:
            return None
        trials += 1
        if matches(guess, algo, target, salt):
            return guess
        return None

    hit = try_plaintext(seed)
    if hit is not None:
        return hit

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

                hit = try_plaintext(ns)
                if hit is not None:
                    return hit
                if trials >= budget:
                    return None

        if not children:
            break
        frontier = heapq.nsmallest(beam_size, children)

    return None


def crack_with_ml(
    model_dir: Path | str,
    target: str,
    algo: str,
    *,
    seed: str,
    beam_size: int,
    max_length: int,
    budget: int,
    salt: str | None = None,
) -> str | None:
    model, meta = load_model_bundle(model_dir)
    return crack_with_ml_bundle(
        model,
        meta,
        target,
        algo,
        seed=seed,
        beam_size=beam_size,
        max_length=max_length,
        budget=budget,
        salt=salt,
    )
