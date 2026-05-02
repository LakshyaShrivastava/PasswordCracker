import hashlib

import pytest

from passcrack.hashing import hash_one, matches, normalize_hex_target


def test_sha256_no_salt():
    assert hash_one("abc123", "sha256") == hashlib.sha256(b"abc123").hexdigest()


def test_sha256_with_salt():
    salt = "pepper"
    expected = hashlib.sha256((salt + "abc123").encode()).hexdigest()
    assert hash_one("abc123", "sha256", salt=salt) == expected


def test_matches_hex():
    h = hash_one("x", "md5")
    assert matches("x", "md5", h)


def test_normalize_hex_target():
    assert normalize_hex_target(" 0xABC ") == "abc"


@pytest.mark.parametrize("algo", ["md5", "sha1", "sha256", "sha512"])
def test_hash_one_algo_roundtrip(algo):
    h = hash_one("test", algo)
    assert matches("test", algo, h)


def test_bcrypt_matches_when_installed():
    bcrypt = pytest.importorskip("bcrypt", reason="bcrypt optional extra")
    stored = bcrypt.hashpw(b"secret", bcrypt.gensalt(rounds=4))
    assert matches("secret", "bcrypt", stored.decode("ascii"))
