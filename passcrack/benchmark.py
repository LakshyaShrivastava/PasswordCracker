"""Fixed-duration hashing throughput benchmarks."""

from __future__ import annotations

import time

from passcrack.hashing import HASHLIB_ALGOS, hash_one

__all__ = ["benchmark_algos", "markdown_table"]


def benchmark_algos(
    algos: list[str],
    duration_sec: float,
    *,
    salt: str | None = None,
    warmup: int = 50,
) -> dict[str, float]:
    """
    Return guesses-per-second for each algorithm over approximately ``duration_sec``.

    For bcrypt, each iteration is a full ``hashpw`` with a fresh salt — dominated by bcrypt cost;
    numbers are illustrative, not comparable to preimage brute-force on a *fixed* hash.
    """
    results: dict[str, float] = {}
    for raw in algos:
        algo = raw.lower().strip()
        if algo == "bcrypt":
            try:
                import bcrypt  # type: ignore[import-untyped]
            except ImportError:
                results[algo] = float("nan")
                continue

            warm = min(2, warmup)
            for i in range(warm):
                bcrypt.hashpw(f"w{i}".encode(), bcrypt.gensalt(rounds=12))

            count = 0
            t0 = time.perf_counter()
            i = 0
            while time.perf_counter() - t0 < duration_sec:
                bcrypt.hashpw(f"g{i}".encode(), bcrypt.gensalt(rounds=12))
                count += 1
                i += 1
            elapsed = time.perf_counter() - t0
            results[algo] = count / elapsed if elapsed > 0 else float("nan")
            continue

        if algo not in HASHLIB_ALGOS:
            raise ValueError(f"Unknown algo {raw!r}")

        for i in range(warmup):
            hash_one(f"w{i}", algo, salt)

        count = 0
        t0 = time.perf_counter()
        i = 0
        while time.perf_counter() - t0 < duration_sec:
            hash_one(f"g{i}", algo, salt)
            count += 1
            i += 1
        elapsed = time.perf_counter() - t0
        results[algo] = count / elapsed if elapsed > 0 else float("nan")

    return results


def markdown_table(results: dict[str, float], *, duration_sec: float) -> str:
    lines = [
        f"Throughput over ~**{duration_sec:g}s** wall clock (machine-dependent).",
        "",
        "| Algorithm | Guesses / sec (approx.) |",
        "|-----------|-------------------------|",
    ]
    for algo in sorted(results.keys()):
        r = results[algo]
        cell = "—" if r != r else f"{r:,.0f}"
        lines.append(f"| `{algo}` | {cell} |")
    lines.append("")
    lines.append(
        "*Interpretation:* MD-family digests are cheap per guess; **bcrypt** is intentionally slow "
        "(here `gensalt(rounds=12)` per guess). Real systems use per-password salts and adaptive "
        "work factors so offline cracking does not scale like raw SHA-256."
    )
    return "\n".join(lines)
