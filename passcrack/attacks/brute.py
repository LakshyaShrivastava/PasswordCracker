"""Bounded brute-force enumeration over a charset."""

from __future__ import annotations

import itertools
import string
from collections.abc import Iterator

__all__ = ["CHARSETS", "charset_from_preset", "iter_brute", "crack_brute"]

CHARSETS = {
    "lower": string.ascii_lowercase,
    "upper": string.ascii_uppercase,
    "digit": string.digits,
    "alphanum": string.ascii_letters + string.digits,
    "lowerdigit": string.ascii_lowercase + string.digits,
}


def charset_from_preset(name: str) -> str:
    key = name.lower().strip()
    if key not in CHARSETS:
        raise ValueError(f"Unknown charset preset {name!r}. Choose: {sorted(CHARSETS)}")
    return CHARSETS[key]


def iter_brute(charset: str, max_length: int) -> Iterator[str]:
    if max_length < 1:
        return
    for length in range(1, max_length + 1):
        for tup in itertools.product(charset, repeat=length):
            yield "".join(tup)


def crack_brute(
    target: str,
    algo: str,
    *,
    charset: str,
    max_length: int,
    salt: str | None = None,
    budget: int | None = None,
) -> str | None:
    """Return first plaintext match or None. ``budget`` caps guesses (optional)."""
    from passcrack.hashing import matches

    count = 0
    for guess in iter_brute(charset, max_length):
        if matches(guess, algo, target, salt):
            return guess
        count += 1
        if budget is not None and count >= budget:
            return None
    return None
