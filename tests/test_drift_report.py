from credit_monitoring.drift.report import build_drift_summary_markdown, classify_severity


def test_classify_severity_categorical_psi():
    assert classify_severity("cat", 0.3) == "severo"
    assert classify_severity("cat", 0.15) == "moderado"
    assert classify_severity("cat", 0.02) == "leve"


def test_classify_severity_numeric_ks_pvalue():
    assert classify_severity("num", 0.0001) == "severo"
    assert classify_severity("num", 0.005) == "moderado"
    assert classify_severity("num", 0.5) == "leve"


def test_build_drift_summary_markdown_lists_drifted_columns():
    summary = {
        "dataset_drift_detected": True,
        "share_of_drifted_columns": 0.2,
        "number_of_drifted_columns": 2,
        "number_of_columns": 10,
        "drifted_columns": [
            {
                "column": "income",
                "column_type": "num",
                "stattest_name": "K-S p_value",
                "drift_score": 1.2e-20,
            },
            {
                "column": "employment_years",
                "column_type": "num",
                "stattest_name": "K-S p_value",
                "drift_score": 5e-50,
            },
        ],
    }

    markdown = build_drift_summary_markdown(summary)

    assert "income" in markdown
    assert "employment_years" in markdown
    assert "severo" in markdown
    assert "SIM" in markdown


def test_build_drift_summary_markdown_no_drift():
    summary = {
        "dataset_drift_detected": False,
        "share_of_drifted_columns": 0.0,
        "number_of_drifted_columns": 0,
        "number_of_columns": 10,
        "drifted_columns": [],
    }

    markdown = build_drift_summary_markdown(summary)

    assert "Nenhuma feature apresentou drift" in markdown
