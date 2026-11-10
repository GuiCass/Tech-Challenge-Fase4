"""Gera o dataset de referencia (limpo) a partir do CSV bruto do Home Credit.

Seleciona um subconjunto de colunas relevantes para o problema de risco de
credito, limpa valores invalidos/anomalos e produz uma amostra estratificada
(por TARGET) de tamanho controlado, para manter o repositorio leve.

Uso:
    poetry run python -m credit_monitoring.data.preprocess
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_monitoring.utils.paths import (
    RAW_APPLICATION_FILE,
    REFERENCE_DIR,
    REFERENCE_FILE,
)

logger = logging.getLogger(__name__)

# DAYS_EMPLOYED usa 365243 como valor-sentinela para "nao empregado"/aposentado.
EMPLOYED_ANOMALY_DAYS = 365243

RAW_COLUMNS = {
    "SK_ID_CURR": "client_id",
    "TARGET": "target",
    "CODE_GENDER": "gender",
    "CNT_CHILDREN": "num_children",
    "AMT_INCOME_TOTAL": "income",
    "AMT_CREDIT": "credit_amount",
    "AMT_ANNUITY": "annuity_amount",
    "NAME_EDUCATION_TYPE": "education",
    "NAME_FAMILY_STATUS": "family_status",
    "NAME_HOUSING_TYPE": "housing_type",
    "DAYS_BIRTH": "age",
    "DAYS_EMPLOYED": "employment_years",
    "FLAG_OWN_CAR": "owns_car",
    "FLAG_OWN_REALTY": "owns_realty",
    "OCCUPATION_TYPE": "occupation_type",
    "REGION_RATING_CLIENT": "region_rating",
    "ORGANIZATION_TYPE": "organization_type",
}

SAMPLE_SIZE = 25_000
RANDOM_STATE = 42


def _convert_age_and_employment(df: pd.DataFrame) -> pd.DataFrame:
    df["age"] = (-df["age"] / 365.25).round(1)

    is_not_employed = df["employment_years"] == EMPLOYED_ANOMALY_DAYS
    df.loc[is_not_employed, "employment_years"] = 0
    df.loc[~is_not_employed, "employment_years"] = (
        -df.loc[~is_not_employed, "employment_years"] / 365.25
    )
    df["employment_years"] = df["employment_years"].clip(lower=0).round(1)
    return df


def build_reference_dataset(
    raw_path=RAW_APPLICATION_FILE,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Le o CSV bruto, limpa e retorna a amostra de referencia."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} nao encontrado. Rode "
            "'poetry run python -m credit_monitoring.data.download' "
            "ou baixe manualmente antes de pre-processar."
        )

    logger.info("Lendo %s...", raw_path)
    df = pd.read_csv(raw_path, usecols=list(RAW_COLUMNS.keys()))
    df = df.rename(columns=RAW_COLUMNS)

    df = _convert_age_and_employment(df)

    # Regras de limpeza do dataset de REFERENCIA (deve passar no contrato da Etapa 1).
    df = df.drop_duplicates(subset="client_id")
    df = df.dropna(subset=["income", "age", "target"])
    df = df[(df["age"] > 18) & (df["age"] < 100)]
    df = df[df["income"] > 0]

    logger.info("Dataset limpo: %d linhas restantes.", len(df))

    if len(df) > sample_size:
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df["target"],
            random_state=random_state,
        )

    df = df.sort_values("client_id").reset_index(drop=True)
    logger.info("Amostra de referencia final: %d linhas.", len(df))
    return df


def save_reference_dataset(df: pd.DataFrame, out_path=REFERENCE_FILE) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Dataset de referencia salvo em %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reference_df = build_reference_dataset()
    save_reference_dataset(reference_df)
