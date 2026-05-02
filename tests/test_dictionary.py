import hashlib
from pathlib import Path

from passcrack.attacks.dictionary import crack_dictionary, iter_wordlist


def test_iter_wordlist_skips_comments(tmp_path: Path):
    p = tmp_path / "w.txt"
    p.write_text("alpha\n# skip\nbeta\n", encoding="utf-8")
    assert list(iter_wordlist(p)) == ["alpha", "beta"]


def test_crack_dictionary(tmp_path: Path):
    p = tmp_path / "w.txt"
    p.write_text("secret\nother\n", encoding="utf-8")
    target = hashlib.sha256(b"secret").hexdigest()
    found = crack_dictionary(target, "sha256", p, rules_spec="identity")
    assert found == "secret"
