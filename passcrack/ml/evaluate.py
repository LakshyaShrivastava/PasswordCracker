"""Compare ML-guided search vs dictionary baseline on held-out plaintexts."""

from __future__ import annotations

from pathlib import Path

from passcrack.attacks.dictionary import crack_dictionary
from passcrack.hashing import hash_one
from passcrack.ml.crack import crack_with_ml_bundle
from passcrack.ml.generate import load_model_bundle
from passcrack.ml.data import load_passwords_from_file

__all__ = ["evaluate_recovery", "report_markdown"]


def evaluate_recovery(
    model_dir: Path | str,
    holdout_path: Path | str,
    *,
    algo: str,
    salt: str | None,
    baseline_wordlist: Path | str,
    budget: int,
    beam_size: int,
    max_length: int,
) -> dict[str, float | int]:
    algo_l = algo.lower().strip()
    if algo_l == "bcrypt":
        raise ValueError(
            "evaluate() supports hashlib digests only: each bcrypt hash embeds a random salt, "
            "so you must compare against precomputed targets rather than hash_one() per row."
        )

    holdout = load_passwords_from_file(holdout_path)
    if not holdout:
        raise ValueError("Holdout wordlist is empty")

    model, meta = load_model_bundle(model_dir)

    ml_hits = 0
    dict_hits = 0

    for pwd in holdout:
        target = hash_one(pwd, algo_l, salt)
        seed = pwd[0] if pwd else ""

        if crack_with_ml_bundle(
            model,
            meta,
            target,
            algo_l,
            seed=seed,
            beam_size=beam_size,
            max_length=max_length,
            budget=budget,
            salt=salt,
        ):
            ml_hits += 1

        if crack_dictionary(
            target,
            algo_l,
            baseline_wordlist,
            rules_spec="identity",
            salt=salt,
            budget=budget,
        ):
            dict_hits += 1

    n = len(holdout)
    return {
        "n": n,
        "ml_hits": ml_hits,
        "dict_hits": dict_hits,
        "ml_rate": ml_hits / n,
        "dict_rate": dict_hits / n,
        "budget": budget,
    }


def report_markdown(stats: dict[str, float | int]) -> str:
    return "\n".join(
        [
            "## ML vs dictionary baseline (held-out)",
            "",
            f"- Passwords evaluated: **{stats['n']}**",
            f"- Guess budget per hash: **{stats['budget']}**",
            "",
            "| Method | Recovered | Rate |",
            "|--------|-----------|------|",
            f"| ML beam | {stats['ml_hits']} | {stats['ml_rate']:.1%} |",
            f"| Dictionary (identity only, baseline wordlist) | {stats['dict_hits']} | {stats['dict_rate']:.1%} |",
            "",
            "*Seeds for ML rows use the first character of each held-out password when present "
            "(optimistic — real attackers rarely know plaintext prefixes).*",
        ]
    )
