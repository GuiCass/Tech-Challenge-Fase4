import joblib
import pandas as pd

from credit_monitoring.model.features import train_test_split_reference
from credit_monitoring.model.train import build_model_pipeline, evaluate


def test_build_model_pipeline_fits_and_predicts(synthetic_reference_df):
    X_train, X_test, y_train, y_test = train_test_split_reference(
        synthetic_reference_df, test_size=0.2, random_state=0
    )

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    assert len(predictions) == len(X_test)
    assert probabilities.shape == (len(X_test), 2)
    assert set(predictions).issubset({0, 1})


def test_evaluate_returns_expected_metric_keys(synthetic_reference_df):
    X_train, X_test, y_train, y_test = train_test_split_reference(
        synthetic_reference_df, test_size=0.2, random_state=0
    )
    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)

    expected_keys = {
        "accuracy",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
        "n_test_samples",
    }
    assert expected_keys == set(metrics.keys())
    assert metrics["n_test_samples"] == len(X_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_trained_pipeline_is_serializable(synthetic_reference_df, tmp_path):
    X_train, _, y_train, _ = train_test_split_reference(synthetic_reference_df)
    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    loaded_pipeline = joblib.load(model_path)

    sample = X_train.iloc[:5]
    pd.testing.assert_frame_equal(
        pd.DataFrame(pipeline.predict_proba(sample)),
        pd.DataFrame(loaded_pipeline.predict_proba(sample)),
    )
