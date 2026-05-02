# PasswordCracker

Small **Python educational** demos around **password hashing** and **sequence modeling**:

- **`algorithms.py` / `main.py`** — given a **SHA-256** hex digest, brute-force guesses over a bounded character set (`a-z`, `0-9`) up to a maximum length (`brute_force(..., max_length)`). The default `main` path exercises the function on a known hash of `abc123`.
- **`ML_cracker.py`** — toy **TensorFlow/Keras** **LSTM** that learns character transitions from a tiny synthetic password list and generates continuations from a seed (illustrative only; not a serious password-recovery tool).

This repository exists for **coursework, interviews, and controlled lab study** — not for attacking real accounts or systems.

## Ethics and legal use

**Use only where you have explicit authorization** (your own hashes, designated CTF/lab targets, coursework sandboxes). Cracking or guessing credentials without permission is illegal in most jurisdictions and against the policies of virtually every platform. If you reference this repo publicly, keep that framing explicit.

## Requirements

- **Python 3.10+** recommended  
- Standard library for the brute-force path  
- TensorFlow **2.x** for `ML_cracker.py` (install when you run that script)

Install TensorFlow when needed, for example:

```bash
pip install tensorflow
```

## How to run

**Brute-force demo (SHA-256 over `a-z` + digits):**

```bash
python main.py
```

**LSTM toy trainer:**

```bash
python ML_cracker.py
```

**Complexity note:** brute-force cost grows exponentially with alphabet size and maximum length; small `max_length` values complete quickly, larger ones become prohibitive — which is exactly why slow password hashing algorithms and salting matter in production systems.

## Testing / misc

`test.py` is an unrelated minimal Keras smoke snippet left from experimentation; it is not required for `main.py`.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
