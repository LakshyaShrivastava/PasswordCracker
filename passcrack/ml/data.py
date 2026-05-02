"""Vocabulary, padding, and training pair construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["Meta", "build_char_maps", "encode_prefix", "make_training_pairs", "load_passwords_from_file"]


def load_passwords_from_file(path: Path | str) -> list[str]:
    lines: list[str] = []
    with Path(path).open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def build_char_maps(passwords: list[str]) -> dict[str, int]:
    """Map each character to 1..V (reserve 0 for padding)."""
    chars = sorted({c for p in passwords for c in p})
    if not chars:
        raise ValueError("No characters inferred from password list")
    return {c: i + 1 for i, c in enumerate(chars)}


def encode_prefix(prefix: str, max_length: int, char_to_int: dict[str, int]) -> list[int]:
    ids = [char_to_int[c] for c in prefix]
    if len(ids) > max_length:
        ids = ids[-max_length:]
    pad = max_length - len(ids)
    return [0] * pad + ids


def make_training_pairs(
    passwords: list[str],
    max_length: int,
    char_to_int: dict[str, int],
) -> tuple[list[list[int]], list[int]]:
    xs: list[list[int]] = []
    ys: list[int] = []
    for pwd in passwords:
        if not pwd:
            continue
        trimmed = pwd[:max_length]
        for i in range(len(trimmed)):
            prefix = trimmed[:i]
            next_ch = trimmed[i]
            xs.append(encode_prefix(prefix, max_length, char_to_int))
            ys.append(char_to_int[next_ch])
    if not xs:
        raise ValueError("No training pairs produced (passwords too short?)")
    return xs, ys


class Meta(dict[str, Any]):
    """Serialized alongside the Keras model."""

    def __init__(
        self,
        *,
        max_input_length: int,
        char_to_int: dict[str, int],
        pad_idx: int = 0,
    ) -> None:
        super().__init__(
            version=1,
            max_input_length=max_input_length,
            char_to_int=char_to_int,
            pad_idx=pad_idx,
            vocab_size=len(char_to_int) + 1,
        )

    @property
    def int_to_char(self) -> dict[int, str]:
        return {v: k for k, v in self["char_to_int"].items()}

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(dict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> Meta:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            max_input_length=raw["max_input_length"],
            char_to_int=dict(raw["char_to_int"]),
            pad_idx=int(raw.get("pad_idx", 0)),
        )
