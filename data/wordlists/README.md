# Bundled wordlists

- **`common-passwords-tiny.txt`** — ~200 synthetic/common-pattern passwords authored for this repo (safe to redistribute under the project Apache-2.0 license).
- **`holdout.txt`** — disjoint random-ish strings used only for `passcrack ml evaluate` demos.

For larger experiments, fetch SecLists (MIT) via [`scripts/fetch_seclists.py`](../../scripts/fetch_seclists.py) into `data/wordlists/seclists/` (gitignored).
