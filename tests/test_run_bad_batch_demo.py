import run_bad_batch_demo as demo

from credit_monitoring.data.contracts import validation_report


def test_build_bad_batch_violates_expected_rules(synthetic_reference_df):
    bad_batch = demo.build_bad_batch(synthetic_reference_df)

    failure_cases = validation_report(bad_batch)
    assert failure_cases is not None

    violated_columns = set(failure_cases["column"].dropna())
    assert {"age", "income", "client_id", "credit_amount"}.issubset(violated_columns)


def test_run_demo_blocks_corrupted_batch_and_writes_report(
    monkeypatch, synthetic_reference_df, tmp_path
):
    report_path = tmp_path / "bad_batch_validation_report.csv"
    monkeypatch.setattr(demo, "load_reference_dataset", lambda: synthetic_reference_df)
    monkeypatch.setattr(demo, "BAD_BATCH_REPORT_FILE", report_path)

    assert demo.run_demo() is True
    assert report_path.exists()

    report_df = demo.pd.read_csv(report_path)
    assert len(report_df) > 0


def test_run_demo_returns_false_if_clean_dataset_is_too_small(monkeypatch, synthetic_reference_df):
    monkeypatch.setattr(demo, "load_reference_dataset", lambda: synthetic_reference_df.head(2))

    try:
        demo.run_demo()
        assert False, "esperava ValueError para dataset pequeno demais"
    except ValueError:
        pass
