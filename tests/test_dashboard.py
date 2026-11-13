import pytest

from credit_monitoring.monitoring.dashboard import build_dashboard_html, save_dashboard


@pytest.fixture
def sample_pipeline_result() -> dict:
    return {
        "training_metrics": {
            "accuracy": 0.81,
            "roc_auc": 0.74,
            "f1": 0.33,
            "n_test_samples": 5000,
        },
        "prediction_comparison": {
            "reference": {"n_samples": 25000, "mean_score": 0.12, "pct_high_risk": 3.5},
            "production": {"n_samples": 25000, "mean_score": 0.19, "pct_high_risk": 8.1},
            "mean_score_delta": 0.07,
            "pct_high_risk_delta": 4.6,
        },
        "drift_summary": {
            "dataset_drift_detected": True,
            "number_of_drifted_columns": 2,
            "number_of_columns": 16,
            "share_of_drifted_columns": 0.125,
            "drifted_columns": [
                {
                    "column": "income",
                    "column_type": "num",
                    "stattest_name": "ks",
                    "drift_score": 0.0001,
                },
                {
                    "column": "employment_years",
                    "column_type": "num",
                    "stattest_name": "ks",
                    "drift_score": 0.0003,
                },
            ],
        },
        "events": [
            {
                "stage": "validacao_dados",
                "status": "success",
                "duration_seconds": 0.5,
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "stage": "treino_modelo",
                "status": "success",
                "duration_seconds": 12.3,
                "timestamp": "2026-01-01T00:00:05+00:00",
            },
        ],
    }


def test_build_dashboard_html_includes_model_metrics(sample_pipeline_result):
    html = build_dashboard_html(sample_pipeline_result)

    assert "0.8100" in html
    assert "5000" in html


def test_build_dashboard_html_includes_score_comparison(sample_pipeline_result):
    html = build_dashboard_html(sample_pipeline_result)

    assert "0.1200" in html
    assert "0.1900" in html
    assert "+0.0700" in html


def test_build_dashboard_html_flags_drift_alert(sample_pipeline_result):
    html = build_dashboard_html(sample_pipeline_result)

    assert "Drift detectado" in html
    assert "income" in html
    assert "employment_years" in html


def test_build_dashboard_html_without_drift_shows_no_alert(sample_pipeline_result):
    sample_pipeline_result["drift_summary"]["dataset_drift_detected"] = False
    sample_pipeline_result["drift_summary"]["drifted_columns"] = []

    html = build_dashboard_html(sample_pipeline_result)

    assert "Nenhum drift significativo" in html
    assert "Drift detectado" not in html


def test_build_dashboard_html_lists_pipeline_events(sample_pipeline_result):
    html = build_dashboard_html(sample_pipeline_result)

    assert "validacao_dados" in html
    assert "treino_modelo" in html


def test_save_dashboard_writes_html_file(tmp_path, sample_pipeline_result):
    out_file = tmp_path / "dashboard.html"

    save_dashboard(sample_pipeline_result, out_path=out_file)

    assert out_file.exists()
    assert "<html" in out_file.read_text(encoding="utf-8")
