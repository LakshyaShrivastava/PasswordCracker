import hashlib

from passcrack.attacks.brute import charset_from_preset, crack_brute, iter_brute


def test_charset_from_preset():
    assert "a" in charset_from_preset("lowerdigit")
    assert "A" in charset_from_preset("alphanum")


def test_iter_brute_short():
    gen = list(iter_brute("ab", max_length=2))
    assert gen[:3] == ["a", "b", "aa"]


def test_crack_brute_finds_short_secret():
    # Keep charset tiny — full lowerdigit^6 is ~2e9 guesses (too slow for unit tests).
    target = hashlib.sha256(b"z").hexdigest()
    found = crack_brute(
        target,
        "sha256",
        charset="abcdefghijklmnopqrstuvwxyz",
        max_length=1,
    )
    assert found == "z"


def test_crack_brute_budget():
    target = hashlib.sha256(b"zzz").hexdigest()
    found = crack_brute(
        target,
        "sha256",
        charset="z",
        max_length=3,
        budget=2,
    )
    assert found is None
