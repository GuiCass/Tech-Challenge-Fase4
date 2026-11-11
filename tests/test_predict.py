import json

import pytest

from credit_monitoring.drift.simulate import simulate_production_dataset
from credit_monitoring.model.features import train_test_split_reference
from credit_monitoring.model.train import build_model_pipeline
import credit_monitoring.model.predict as predict_module
from credit_monitoring.model.predict import (
    compare_reference_and_production_scores,
    generate_predictions,
    summarize_scores,
)


@pytest.fixture
def trained_pipeline(synthetic_reference_df):
    X_train, _, y_train, _ = train_test_split_reference(synthetic_reference_df, test_size=0.2)
    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def test_generate_predictions_has_expected_shape_and_columns(
    trained_pipeline, synthetic_reference_df
):
    predictions = generate_predictions(trained_pipeline, synthetic_reference_df)

    assert len(predictions) == len(synthetic_reference_df)
    assert set(predictions.columns) == {"client_id", "predicted_proba", "predicted_label"}
    assert predictions["predicted_proba"].between(0, 1).all()
    assert set(predictions["predicted_label"].unique()).issubset({0, 1})


def test_summarize_scores_returns_expected_keys(trained_pipeline, synthetic_reference_df):
    predictions = generate_predictions(trained_pipeline, synthetic_reference_df)
    summary = summarize_scores(predictions)

    assert set(summary.keys()) == {"n_samples", "mean_score", "std_score", "pct_high_risk"}
    assert summary["n_samples"] == len(synthetic_reference_df)


def test_compare_reference_and_production_scores(trained_pipeline, synthetic_reference_df):
    production_df = simulate_production_dataset(synthetic_reference_df, random_state=123)

    comparison = compare_reference_and_production_scores(
        trained_pipeline, synthetic_reference_df, production_df
    )

    assert "reference" in comparison and "production" in comparison
    assert "mean_score_delta" in comparison
    assert "pct_high_risk_delta" in comparison


def test_run_predictions_writes_report_and_predictions_file(
    monkeypatch, trained_pipeline, synthetic_reference_df, tmp_path
):
    production_df = simulate_production_dataset(synthetic_reference_df, random_state=123)

    comparison_file = tmp_path / "score_comparison.json"
    predictions_file = tmp_path / "credit_production_predictions.csv"

    monkeypatch.setattr(predict_module, "load_trained_pipeline", lambda: trained_pipeline)
    monkeypatch.setattr(predict_module, "load_reference_dataset", lambda: synthetic_reference_df)
    monkeypatch.setattr(predict_module, "load_production_dataset", lambda: production_df)
    monkeypatch.setattr(predict_module, "SCORE_COMPARISON_FILE", comparison_file)
    monkeypatch.setattr(predict_module, "PRODUCTION_PREDICTIONS_FILE", predictions_file)

    comparison = predict_module.run_predictions()

    assert comparison_file.exists()
    assert predictions_file.exists()
    assert json.loads(comparison_file.read_text()) == comparison
