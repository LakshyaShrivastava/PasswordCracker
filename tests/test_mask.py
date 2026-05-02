import hashlib

import pytest

from passcrack.attacks.mask import charset_for_token, crack_mask, iter_mask, parse_mask


def test_parse_mask_literals_and_tokens():
    assert parse_mask("ab?l?d") == ["a", "b", "?l", "?d"]


def test_iter_mask_small():
    outs = set(iter_mask("?l?d"))
    assert len(outs) == 26 * 10
    assert "a0" in outs and "z9" in outs


def test_charset_for_token_bad():
    with pytest.raises(ValueError):
        charset_for_token("?x")


def test_crack_mask():
    guess = "ab9"
    target = hashlib.md5(guess.encode()).hexdigest()
    found = crack_mask(target, "md5", "?l?l?d")
    assert found == guess
