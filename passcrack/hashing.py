"""Hash algorithms and salted preimage helpers used by all attacks."""

from __future__ import annotations

import hashlib
from typing import Callable

__all__ = [
    "HASHLIB_ALGOS",
    "hash_one",
    "matches",
    "normalize_hex_target",
]


HASHLIB_ALGOS = frozenset({"md5", "sha1", "sha256", "sha512"})


def _salted_bytes(plaintext: str, salt: str | None) -> bytes:
    """Concatenate UTF-8 salt + plaintext (simple educational salted hash)."""
    p = plaintext.encode("utf-8")
    if salt is None or salt == "":
        return p
    return salt.encode("utf-8") + p


def hash_one(plaintext: str, algo: str, salt: str | None = None) -> str:
    """
    Compute a hex digest for hashlib algorithms, or bcrypt hash string.

    ``algo`` is case-insensitive. For bcrypt, ``salt`` must be omitted;
    a random salt is generated via bcrypt defaults when *hashing* for display —
    use ``matches`` for verification against an existing bcrypt hash.
    """
    algo_l = algo.lower().strip()
    if algo_l == "bcrypt":
        try:
            import bcrypt  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("Install bcrypt extra: pip install passcrack-edu[bcrypt]") from e
        hashed = bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt())
        return hashed.decode("ascii")

    if algo_l not in HASHLIB_ALGOS:
        raise ValueError(f"Unsupported algo {algo!r}. Choose one of {sorted(HASHLIB_ALGOS)} or bcrypt.")

    data = _salted_bytes(plaintext, salt)
    h = hashlib.new(algo_l, data)
    return h.hexdigest()


def normalize_hex_target(target: str) -> str:
    t = target.strip().lower()
    if t.startswith("0x"):
        t = t[2:]
    return t


def matches(plaintext: str, algo: str, target: str, salt: str | None = None) -> bool:
    """True if plaintext hashes (or verifies for bcrypt) to ``target``."""
    algo_l = algo.lower().strip()
    if algo_l == "bcrypt":
        try:
            import bcrypt  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("Install bcrypt extra: pip install passcrack-edu[bcrypt]") from e
        return bcrypt.checkpw(plaintext.encode("utf-8"), target.strip().encode("ascii"))

    if algo_l not in HASHLIB_ALGOS:
        raise ValueError(f"Unsupported algo {algo!r}")

    digest = hash_one(plaintext, algo_l, salt)
    return digest == normalize_hex_target(target)


def hash_iterator(algo: str, salt: str | None = None) -> Callable[[str], str]:
    """Return ``lambda p: hash_one(p, algo, salt)`` for benchmarks."""

    def _h(p: str) -> str:
        return hash_one(p, algo, salt)

    return _h
