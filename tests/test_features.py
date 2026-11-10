import numpy as np
import pandas as pd
import pytest

from credit_monitoring.model.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    train_test_split_reference,
)


@pytest.fixture
def synthetic_reference_df() -> pd.DataFrame:
    """Pequeno dataset sintetico com o mesmo schema do dataset de referencia real.

    Usado nos testes para nao depender do download real do Kaggle.
    """
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame(
        {
            "client_id": range(n),
            "target": rng.integers(0, 2, size=n),
            "gender": rng.choice(["M", "F"], size=n),
            "num_children": rng.integers(0, 4, size=n),
            "income": rng.uniform(1_000, 20_000, size=n),
            "credit_amount": rng.uniform(5_000, 100_000, size=n),
            "annuity_amount": rng.uniform(500, 5_000, size=n),
            "education": rng.choice(["Secondary", "Higher education"], size=n),
            "family_status": rng.choice(["Married", "Single"], size=n),
            "housing_type": rng.choice(["House / apartment", "Rented apartment"], size=n),
            "age": rng.uniform(19, 70, size=n),
            "employment_years": rng.uniform(0, 30, size=n),
            "owns_car": rng.choice(["Y", "N"], size=n),
            "owns_realty": rng.choice(["Y", "N"], size=n),
            # inclui NaN de proposito para validar a imputacao
            "occupation_type": rng.choice(["Laborers", "Managers", None], size=n),
            "region_rating": rng.integers(1, 4, size=n),
            "organization_type": rng.choice(["Business Entity", "Self-employed"], size=n),
        }
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
