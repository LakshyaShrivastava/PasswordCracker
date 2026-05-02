from passcrack.benchmark import benchmark_algos, markdown_table


def test_benchmark_md5_runs():
    res = benchmark_algos(["md5"], duration_sec=0.08, warmup=3)
    assert "md5" in res
    assert res["md5"] == res["md5"]  # not nan
    assert res["md5"] > 50  # CI runners vary widely


def test_markdown_table_has_header():
    res = {"md5": 1.234e6}
    md = markdown_table(res, duration_sec=1.0)
    assert "Guesses / sec" in md
    assert "md5" in md
