"""Dictionary attack with optional rule expansion."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from passcrack.attacks.rules import expand_word, parse_rule_names

__all__ = ["iter_wordlist", "crack_dictionary"]


def iter_wordlist(path: Path | str, *, encoding: str = "utf-8") -> Iterator[str]:
    p = Path(path)
    with p.open(encoding=encoding, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line


def crack_dictionary(
    target: str,
    algo: str,
    wordlist: Path | str,
    rules_spec: str | None,
    *,
    salt: str | None = None,
    budget: int | None = None,
) -> str | None:
    from passcrack.hashing import matches

    names = parse_rule_names(rules_spec)
    count = 0
    for word in iter_wordlist(wordlist):
        for guess in expand_word(word, names):
            if matches(guess, algo, target, salt):
                return guess
            count += 1
            if budget is not None and count >= budget:
                return None
    return None
