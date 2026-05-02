#!/usr/bin/env python3
"""
Download a small slice of SecLists (MIT license) for optional offline experiments.

Writes under ``data/wordlists/seclists/`` (gitignored by default).
Requires network access.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://raw.githubusercontent.com/danielmiessler/SecLists/master/"
    "Passwords/Common-Credentials/10k-most-common.txt"
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a SecLists password slice.")
    ap.add_argument("--url", default=DEFAULT_URL, help="Raw text URL (default: SecLists 10k snippet file).")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/wordlists/seclists/10k-most-common.txt"),
        help="Output path.",
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url, timeout=60) as resp:  # noqa: S310
        data = resp.read()
    args.out.write_bytes(data)
    print(f"Wrote {len(data)} bytes to {args.out}")


if __name__ == "__main__":
    main()
