"""Command-line interface for passcrack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from passcrack import __version__
from passcrack.attacks.brute import charset_from_preset, crack_brute
from passcrack.attacks.dictionary import crack_dictionary
from passcrack.attacks.mask import crack_mask
from passcrack.benchmark import benchmark_algos, markdown_table
from passcrack.hashing import hash_one, normalize_hex_target


def _cmd_hash(args: argparse.Namespace) -> int:
    digest = hash_one(args.plaintext, args.algo, args.salt)
    print(digest)
    return 0


def _target_for_algo(algo: str, target: str) -> str:
    a = algo.lower()
    if a == "bcrypt":
        return target.strip()
    return normalize_hex_target(target)


def _cmd_brute(args: argparse.Namespace) -> int:
    charset = charset_from_preset(args.charset)
    target = _target_for_algo(args.algo, args.target)
    found = crack_brute(
        target,
        args.algo.lower(),
        charset=charset,
        max_length=args.max_length,
        salt=args.salt,
        budget=args.budget,
    )
    if found:
        print(f"FOUND: {found}")
        return 0
    print("NOT FOUND (exhausted search space or budget).")
    return 1


def _cmd_dict(args: argparse.Namespace) -> int:
    target = _target_for_algo(args.algo, args.target)
    found = crack_dictionary(
        target,
        args.algo.lower(),
        args.wordlist,
        args.rules,
        salt=args.salt,
        budget=args.budget,
    )
    if found:
        print(f"FOUND: {found}")
        return 0
    print("NOT FOUND (exhausted wordlist variants or budget).")
    return 1


def _cmd_mask(args: argparse.Namespace) -> int:
    target = _target_for_algo(args.algo, args.target)
    found = crack_mask(
        target,
        args.algo.lower(),
        args.mask,
        salt=args.salt,
        budget=args.budget,
    )
    if found:
        print(f"FOUND: {found}")
        return 0
    print("NOT FOUND (exhausted mask space or budget).")
    return 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    res = benchmark_algos(algos, args.duration, salt=args.salt)
    if args.markdown:
        print(markdown_table(res, duration_sec=args.duration))
    else:
        for algo, rate in sorted(res.items()):
            print(f"{algo}\t{rate:,.2f}\tguesses/sec")
    return 0


def _cmd_ml_train(args: argparse.Namespace) -> int:
    from passcrack.ml.train import train_and_save

    train_and_save(
        args.wordlist,
        args.out,
        epochs=args.epochs,
        max_input_length=args.max_input_length,
        validation_split=args.validation_split,
        batch_size=args.batch_size,
    )
    print(f"Wrote model to {Path(args.out) / 'model.keras'} and meta.json")
    return 0


def _cmd_ml_generate(args: argparse.Namespace) -> int:
    from passcrack.ml.generate import beam_candidates, load_model_bundle

    model, meta = load_model_bundle(args.model)
    candidates = beam_candidates(
        model,
        meta,
        seed=args.seed,
        beam_size=args.beam,
        max_length=args.max_length,
        max_candidates=args.limit,
    )
    for s in candidates:
        print(s)
    return 0


def _cmd_ml_crack(args: argparse.Namespace) -> int:
    from passcrack.ml.crack import crack_with_ml

    target = _target_for_algo(args.algo, args.target)
    found = crack_with_ml(
        args.model,
        target,
        args.algo.lower(),
        seed=args.seed,
        beam_size=args.beam,
        max_length=args.max_length,
        budget=args.budget,
        salt=args.salt,
    )
    if found:
        print(f"FOUND: {found}")
        return 0
    print("NOT FOUND within ML candidate budget.")
    return 1


def _cmd_ml_evaluate(args: argparse.Namespace) -> int:
    from passcrack.ml.evaluate import evaluate_recovery, report_markdown

    stats = evaluate_recovery(
        args.model,
        args.holdout,
        algo=args.algo.lower(),
        salt=args.salt,
        baseline_wordlist=args.baseline_wordlist,
        budget=args.budget,
        beam_size=args.beam,
        max_length=args.max_length,
    )
    text = report_markdown(stats)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
        print(f"\nWrote {args.report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="passcrack", description="Educational password hash tooling.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hash", help="Hash a plaintext string (demo / preimage setup).")
    h.add_argument("--algo", required=True, help="md5|sha1|sha256|sha512|bcrypt")
    h.add_argument("--salt", default=None, help="Optional salt prefix (UTF-8) for hashlib digests.")
    h.add_argument("plaintext", help="String to hash.")
    h.set_defaults(func=_cmd_hash)

    b = sub.add_parser("brute", help="Bounded brute-force over a charset preset.")
    b.add_argument("--algo", required=True)
    b.add_argument("--target", required=True, help="Hex digest (hashlib algos).")
    b.add_argument(
        "--charset",
        default="lowerdigit",
        help="Preset: lower|upper|digit|alphanum|lowerdigit (default: lowerdigit).",
    )
    b.add_argument("--max-length", type=int, required=True)
    b.add_argument("--salt", default=None)
    b.add_argument("--budget", type=int, default=None, help="Max guesses (optional).")
    b.set_defaults(func=_cmd_brute)

    d = sub.add_parser("dict", help="Dictionary attack with optional mangling rules.")
    d.add_argument("--algo", required=True)
    d.add_argument("--target", required=True)
    d.add_argument("--wordlist", required=True, type=Path)
    d.add_argument(
        "--rules",
        default="identity",
        help="Comma-separated: identity,capitalize,leet,digit-suffix,digit2-suffix,year-suffix",
    )
    d.add_argument("--salt", default=None)
    d.add_argument("--budget", type=int, default=None)
    d.set_defaults(func=_cmd_dict)

    m = sub.add_parser("mask", help="Mask attack (?l ?u ?d ?s ?a).")
    m.add_argument("--algo", required=True)
    m.add_argument("--target", required=True)
    m.add_argument("--mask", required=True)
    m.add_argument("--salt", default=None)
    m.add_argument("--budget", type=int, default=None)
    m.set_defaults(func=_cmd_mask)

    bench = sub.add_parser("benchmark", help="Measure hashing throughput for ~duration seconds.")
    bench.add_argument("--duration", type=float, default=5.0)
    bench.add_argument("--algos", default="md5,sha1,sha256,sha512", help="Comma-separated list; bcrypt optional.")
    bench.add_argument("--salt", default=None)
    bench.add_argument("--markdown", action="store_true", help="Emit a Markdown table.")
    bench.set_defaults(func=_cmd_benchmark)

    ml = sub.add_parser("ml", help="TensorFlow LSTM candidate generator (optional dependency).")
    ml_sub = ml.add_subparsers(dest="ml_cmd", required=True)

    tr = ml_sub.add_parser("train", help="Train next-char LSTM on a wordlist.")
    tr.add_argument("--wordlist", required=True, type=Path)
    tr.add_argument("--out", required=True, type=Path)
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--max-input-length", type=int, default=16)
    tr.add_argument("--validation-split", type=float, default=0.1)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.set_defaults(func=_cmd_ml_train)

    gen = ml_sub.add_parser("generate", help="Beam-search candidate strings from a trained model.")
    gen.add_argument("--model", required=True, type=Path, help="Directory containing model.keras + meta.json")
    gen.add_argument("--seed", default="")
    gen.add_argument("--beam", type=int, default=8)
    gen.add_argument("--max-length", type=int, default=12)
    gen.add_argument("--limit", type=int, default=50)
    gen.set_defaults(func=_cmd_ml_generate)

    cr = ml_sub.add_parser("crack", help="Score ML-ranked candidates against a target digest.")
    cr.add_argument("--model", required=True, type=Path)
    cr.add_argument("--algo", required=True)
    cr.add_argument("--target", required=True)
    cr.add_argument("--seed", default="")
    cr.add_argument("--beam", type=int, default=8)
    cr.add_argument("--max-length", type=int, default=12)
    cr.add_argument("--budget", type=int, default=10_000)
    cr.add_argument("--salt", default=None)
    cr.set_defaults(func=_cmd_ml_crack)

    ev = ml_sub.add_parser("evaluate", help="Held-out recovery rates vs dictionary baseline.")
    ev.add_argument("--model", required=True, type=Path)
    ev.add_argument("--holdout", required=True, type=Path)
    ev.add_argument("--baseline-wordlist", required=True, type=Path)
    ev.add_argument("--algo", default="sha256")
    ev.add_argument("--salt", default=None)
    ev.add_argument("--budget", type=int, default=2000)
    ev.add_argument("--beam", type=int, default=8)
    ev.add_argument("--max-length", type=int, default=16)
    ev.add_argument("--report", default=None, help="Write Markdown summary to this path.")
    ev.set_defaults(func=_cmd_ml_evaluate)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
