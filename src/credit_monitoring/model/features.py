"""Definicao das features do modelo baseline e utilitarios de split.

Mantem num unico lugar quais colunas do dataset de referencia entram no
modelo, como sao tratadas (imputacao + encoding) e como o dataset e
particionado em treino/teste — para que `train.py` (Etapa 1) e `predict.py`
(Etapa 2) reutilizem exatamente a mesma definicao.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_monitoring.utils.paths import REFERENCE_FILE

ID_COLUMN = "client_id"
TARGET_COLUMN = "target"

NUMERIC_FEATURES = [
    "num_children",
    "income",
    "credit_amount",
    "annuity_amount",
    "age",
    "employment_years",
    "region_rating",
]

CATEGORICAL_FEATURES = [
    "gender",
    "education",
    "family_status",
    "housing_type",
    "owns_car",
    "owns_realty",
    "occupation_type",
    "organization_type",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_reference_dataset(path=REFERENCE_FILE) -> pd.DataFrame:
    """Carrega o dataset de referencia gerado por `data/preprocess.py`."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nao encontrado. Rode "
            "'poetry run python -m credit_monitoring.data.download' e "
            "'poetry run python -m credit_monitoring.data.preprocess' antes."
        )
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer com imputacao + encoding para as features do modelo."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def train_test_split_reference(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Separa X/y e aplica split estratificado por `target`.

    Retorna (X_train, X_test, y_train, y_test), onde X contem apenas as
    colunas de `FEATURE_COLUMNS` (sem o identificador do cliente).
    """
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset de referencia sem as colunas esperadas: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
