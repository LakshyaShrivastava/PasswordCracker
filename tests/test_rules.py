from passcrack.attacks.rules import expand_word, parse_rule_names


def test_parse_prepends_identity():
    names = parse_rule_names("capitalize")
    assert names[0] == "identity"
    assert "capitalize" in names


def test_expand_capitalize_and_leet():
    words = list(expand_word("Password", parse_rule_names("capitalize,leet")))
    assert "Password" in words  # identity
    assert "PASSWORD" in words  # upper branch from capitalize rule
    assert "P455w0rd" in words  # leet
