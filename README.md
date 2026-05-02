# PasswordCracker

Educational **Python** tooling for **password hashing**, **offline preimage demos** (you already have the digest), **throughput benchmarks**, and an optional **TensorFlow LSTM** that ranks candidate passwords with **beam search**.

This repository exists for **coursework, interviews, and controlled lab study** — not for attacking real accounts or systems.

## Ethics and legal use

**Use only where you have explicit authorization** (your own hashes, designated CTF/lab targets, coursework sandboxes). Cracking or guessing credentials without permission is illegal in most jurisdictions and against the policies of virtually every platform. If you reference this repo publicly, keep that framing explicit.

## Features

| Area | What you get |
|------|----------------|
| **CLI** | `passcrack` — `hash`, `brute`, `dict`, `mask`, `benchmark`, `ml …` |
| **Hashes** | `md5`, `sha1`, `sha256`, `sha512`, optional **`bcrypt`** extra |
| **Salt** | Hashlib digests use `SHA*(salt ‖ password)` UTF-8 bytes (simple demo salt; not PBKDF2/Argon2). |
| **Attacks** | Bounded brute-force (charset presets), dictionary + tiny rule engine, mask (`?l?u?d?s?a`). |
| **ML** | Train a next-character **LSTM**, generate candidates with **beam search**, compare recovery vs dictionary baseline (`passcrack ml evaluate`). |

Out of scope: hashcat/John parity, rainbow tables, networking, GPU rules.

## Install

```bash
cd PasswordCracker
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -e ".[dev]"        # pytest + editable install
```

Optional extras:

```bash
pip install -e ".[bcrypt]"     # bcrypt hash + benchmark
pip install -e ".[ml]"         # TensorFlow / Keras stack
```

Console script: **`passcrack`** (also `python main.py`).

## Quick examples

**Fingerprint a digest (SHA-256, salted demo):**

```bash
passcrack hash --algo sha256 --salt demo abc123
```

**Brute-force tiny spaces (always keep charset + length bounds tiny):**

```bash
passcrack brute --algo sha256 --charset lower --max-length 1 --target <hex-digest>
```

**Dictionary + rules:**

```bash
passcrack dict --algo sha256 --target <hex> ^
  --wordlist data/wordlists/common-passwords-tiny.txt ^
  --rules identity,capitalize,leet,year-suffix
```

**Mask attack (`?l` lower, `?u` upper, `?d` digit, `?s` symbol, `?a` alnum):**

```bash
passcrack mask --algo md5 --target <hex> --mask "?l?l?l?d?d"
```

**Benchmark cheap digests (numbers vary by CPU):**

```bash
passcrack benchmark --duration 2 --algos md5,sha1,sha256,sha512 --markdown
```

Example output (laptop-class CPU, illustrative):

```
Throughput over ~**2s** wall clock (machine-dependent).

| Algorithm | Guesses / sec (approx.) |
|-----------|-------------------------|
| `md5` | ~4e5 |
| `sha1` | ~3.5e5 |
| `sha256` | ~3.7e5 |
| `sha512` | ~3e5 |
```

```bash
pip install -e ".[bcrypt]"
passcrack benchmark --duration 3 --algos md5,bcrypt --markdown
```

bcrypt rows show **hashes/sec** using `gensalt(rounds=12)` each guess — intentionally tiny compared to MD5/SHA-256, which is the pedagogical point.

## ML workflow

```bash
pip install -e ".[ml]"
passcrack ml train \
  --wordlist data/wordlists/common-passwords-tiny.txt \
  --out experiments/ml/ckpt \
  --epochs 12 \
  --max-input-length 16

passcrack ml generate --model experiments/ml/ckpt --seed abc --beam 8 --max-length 12 --limit 20

passcrack ml crack --model experiments/ml/ckpt --algo sha256 --target <hex> --seed a --budget 2000

passcrack ml evaluate \
  --model experiments/ml/ckpt \
  --holdout data/wordlists/holdout.txt \
  --baseline-wordlist data/wordlists/common-passwords-tiny.txt \
  --budget 4000 \
  --report experiments/ml_eval_report.md
```

See [`experiments/ml_eval_report.md`](experiments/ml_eval_report.md) for the latest summarized metrics + regeneration commands.

## Repository layout

```
passcrack/           # Python package (CLI + attacks + ML)
data/wordlists/      # Tiny bundled lists + README
scripts/             # Optional SecLists fetch helper
tests/               # pytest suite
experiments/         # ML eval report (checkpoint dir gitignored)
```

## Testing & CI

```bash
pytest tests/ -q --ignore=tests/test_ml_smoke.py   # core tests
pytest tests/test_ml_smoke.py                      # needs TensorFlow
```

GitHub Actions runs the core suite on Python **3.10–3.12** and an optional ML smoke job.

## Wordlists & licensing

Bundled lists are documented in [`data/wordlists/README.md`](data/wordlists/README.md).  
Optional larger corpora: [`scripts/fetch_seclists.py`](scripts/fetch_seclists.py) (SecLists, MIT).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
