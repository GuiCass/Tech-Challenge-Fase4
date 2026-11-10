import numpy as np
import pytest

from credit_monitoring.model.features import (
    FEATURE_COLUMNS,
    build_preprocessor,
    train_test_split_reference,
)


def test_train_test_split_reference_keeps_only_feature_columns(synthetic_reference_df):
    X_train, X_test, y_train, y_test = train_test_split_reference(
        synthetic_reference_df, test_size=0.25, random_state=0
    )

    assert set(X_train.columns) == set(FEATURE_COLUMNS)
    assert "client_id" not in X_train.columns
    assert len(X_train) + len(X_test) == len(synthetic_reference_df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_train_test_split_reference_missing_column_raises(synthetic_reference_df):
    broken_df = synthetic_reference_df.drop(columns=["income"])
    with pytest.raises(ValueError, match="income"):
        train_test_split_reference(broken_df)


def test_build_preprocessor_handles_missing_values(synthetic_reference_df):
    X = synthetic_reference_df[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    # garante que o NaN proposital em occupation_type nao quebra a transformacao
    assert not np.isnan(transformed.toarray() if hasattr(transformed, "toarray") else transformed).any()
