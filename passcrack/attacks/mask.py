"""Hashcat-style lite masks (?l ?u ?d ?s ?a)."""

from __future__ import annotations

import itertools
import string
from collections.abc import Iterator

__all__ = ["charset_for_token", "crack_mask", "iter_mask", "parse_mask"]

_LOWER = string.ascii_lowercase
_UPPER = string.ascii_uppercase
_DIGIT = string.digits
_SYMBOL = "!@#$%^&*()-_=+[]{}|;:,.<>?"
_ALNUM = _LOWER + _UPPER + _DIGIT


def charset_for_token(tok: str) -> str:
    if tok == "?l":
        return _LOWER
    if tok == "?u":
        return _UPPER
    if tok == "?d":
        return _DIGIT
    if tok == "?s":
        return _SYMBOL
    if tok == "?a":
        return _ALNUM
    raise ValueError(f"Unknown mask token {tok!r}")


def parse_mask(mask: str) -> list[str]:
    """
    Split ``mask`` into single-char literals or two-char placeholders starting with '?'.

    Example: ``ab?l?d`` -> ``['a','b','?l','?d']``
    """
    if not mask:
        return []
    i = 0
    parts: list[str] = []
    n = len(mask)
    while i < n:
        if mask[i] == "?" and i + 1 < n:
            parts.append(mask[i : i + 2])
            i += 2
        else:
            parts.append(mask[i])
            i += 1
    return parts


def iter_mask(mask: str) -> Iterator[str]:
    """Lazily enumerate all strings matching ``mask``."""
    parts = parse_mask(mask)
    charsets: list[str] = []
    for p in parts:
        if len(p) == 1:
            charsets.append(p)
        else:
            charsets.append(charset_for_token(p))
    for tup in itertools.product(*charsets):
        yield "".join(tup)


def crack_mask(
    target: str,
    algo: str,
    mask: str,
    *,
    salt: str | None = None,
    budget: int | None = None,
) -> str | None:
    """Try every candidate matching ``mask`` until digest matches ``target``."""
    from passcrack.hashing import matches

    count = 0
    for guess in iter_mask(mask):
        if matches(guess, algo, target, salt):
            return guess
        count += 1
        if budget is not None and count >= budget:
            return None
    return None
