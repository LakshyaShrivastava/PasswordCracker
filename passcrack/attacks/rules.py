"""Tiny composable password-mangling rules for dictionary attacks."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator

__all__ = ["RULE_REGISTRY", "expand_word", "parse_rule_names"]

_LEET_MAP = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})


def _identity(w: str) -> Iterator[str]:
    yield w


def _capitalize(w: str) -> Iterator[str]:
    if not w:
        return
    yield w.capitalize()
    yield w.upper()


def _leet(w: str) -> Iterator[str]:
    yield w.translate(_LEET_MAP)


def _append_digit(w: str) -> Iterator[str]:
    for d in "0123456789":
        yield w + d


def _append_two_digits(w: str) -> Iterator[str]:
    for a, b in itertools.product("0123456789", repeat=2):
        yield w + a + b


def _year_suffix(w: str) -> Iterator[str]:
    for year in range(1990, 2031):
        yield w + str(year)


RULE_REGISTRY: dict[str, callable[[str], Iterator[str]]] = {
    "identity": _identity,
    "capitalize": _capitalize,
    "leet": _leet,
    "digit-suffix": _append_digit,
    "digit2-suffix": _append_two_digits,
    "year-suffix": _year_suffix,
}


def parse_rule_names(spec: str | None) -> list[str]:
    if not spec or not spec.strip():
        return ["identity"]
    names = [x.strip().lower() for x in spec.split(",") if x.strip()]
    out: list[str] = []
    for n in names:
        if n not in RULE_REGISTRY:
            raise ValueError(f"Unknown rule {n!r}. Known: {sorted(RULE_REGISTRY)}")
        out.append(n)
    if "identity" not in out:
        out.insert(0, "identity")
    return out


def expand_word(word: str, rule_names: Iterable[str]) -> Iterator[str]:
    """Yield variants for ``word`` applying each named rule (union, dedup order-preserving)."""
    seen: set[str] = set()
    for name in rule_names:
        for variant in RULE_REGISTRY[name](word):
            if variant not in seen:
                seen.add(variant)
                yield variant
