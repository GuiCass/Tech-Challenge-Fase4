import json

import credit_monitoring.drift.detect as detect_module
from credit_monitoring.drift.detect import run_drift_detection
from credit_monitoring.drift.simulate import simulate_production_dataset


def test_run_drift_detection_flags_income_and_employment_years(
    monkeypatch, synthetic_reference_df, tmp_path
):
    production_df = simulate_production_dataset(synthetic_reference_df, random_state=123)

    monkeypatch.setattr(detect_module, "DRIFT_REPORTS_DIR", tmp_path)
    monkeypatch.setattr(detect_module, "DRIFT_REPORT_HTML_FILE", tmp_path / "drift_report.html")
    monkeypatch.setattr(detect_module, "DRIFT_SUMMARY_FILE", tmp_path / "drift_summary.json")
    monkeypatch.setattr(detect_module, "DRIFT_SUMMARY_MARKDOWN_FILE", tmp_path / "drift_summary.md")

    summary = run_drift_detection(synthetic_reference_df, production_df)

    drifted_names = {c["column"] for c in summary["drifted_columns"]}
    assert {"income", "employment_years"}.issubset(drifted_names)

    assert (tmp_path / "drift_report.html").exists()
    assert (tmp_path / "drift_summary.json").exists()
    assert json.loads((tmp_path / "drift_summary.json").read_text()) == summary

    markdown = (tmp_path / "drift_summary.md").read_text()
    assert "income" in markdown
    assert "employment_years" in markdown


def test_run_drift_detection_reports_no_drift_for_identical_datasets(
    monkeypatch, synthetic_reference_df, tmp_path
):
    monkeypatch.setattr(detect_module, "DRIFT_REPORTS_DIR", tmp_path)
    monkeypatch.setattr(detect_module, "DRIFT_REPORT_HTML_FILE", tmp_path / "drift_report.html")
    monkeypatch.setattr(detect_module, "DRIFT_SUMMARY_FILE", tmp_path / "drift_summary.json")
    monkeypatch.setattr(detect_module, "DRIFT_SUMMARY_MARKDOWN_FILE", tmp_path / "drift_summary.md")

    summary = run_drift_detection(synthetic_reference_df, synthetic_reference_df.copy())

    assert summary["drifted_columns"] == []
    assert summary["dataset_drift_detected"] is False
