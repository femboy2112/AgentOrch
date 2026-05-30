from scripts.cloud_eval import run_test_counts


def test_run_test_counts_reports_fractional_progress_without_x() -> None:
    code = "def f(x):\n    return x\n"
    test_src = """
    from solution import f

    def test_passes():
        assert f(1) == 1

    def test_fails():
        assert f(2) == 99
    """

    ok, _tail, passed_cases, total_cases = run_test_counts(code, test_src, stop_on_first=False)

    assert ok is False
    assert passed_cases == 1
    assert total_cases == 2
