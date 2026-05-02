"""Backward-compatible entry point — prefer the ``passcrack`` console script."""

from passcrack.cli import main

if __name__ == "__main__":
    main()
