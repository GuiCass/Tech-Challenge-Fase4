"""Simula o "Dataset de Producao", representando a passagem do tempo e a
degradacao economica (inflacao, desemprego) sobre a populacao de clientes.

Duas variaveis sofrem drift intencional, conforme pedido no desafio:

1. `income` (renda): uma fracao dos clientes tem a renda real corroida pela
   inflacao (nao acompanha o custo de vida) — simula o cenario de inflacao
   citado no contexto do projeto.
2. `employment_years` (tempo de emprego): uma fracao maior de clientes perdeu
   o emprego recentemente (tempo de emprego perto de zero), simulando alta na
   taxa de desemprego.

O dataset de producao resultante continua satisfazendo o contrato de
qualidade de dados (Etapa 1) — o que muda e a DISTRIBUICAO das variaveis, nao
a validade estrutural dos dados. A deteccao desse tipo de mudanca (Data
Drift) e feita na Etapa 2.3 com o Evidently AI.

Uso:
    poetry run python -m credit_monitoring.drift.simulate
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from credit_monitoring.model.features import load_reference_dataset
from credit_monitoring.utils.paths import PRODUCTION_DIR, PRODUCTION_FILE

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

INCOME_EROSION_FRACTION = 0.6
INCOME_EROSION_FACTOR_RANGE = (0.70, 0.90)

UNEMPLOYMENT_SHOCK_FRACTION = 0.25
UNEMPLOYMENT_SHOCK_YEARS_RANGE = (0.0, 0.5)


def _apply_income_erosion(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Reduz a renda real de uma fracao dos clientes (efeito da inflacao)."""
    affected = rng.random(len(df)) < INCOME_EROSION_FRACTION
    factors = rng.uniform(*INCOME_EROSION_FACTOR_RANGE, size=len(df))
    df.loc[affected, "income"] = (df.loc[affected, "income"] * factors[affected]).round(2)
    return df


def _apply_unemployment_shock(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Zera (ou quase) o tempo de emprego de uma fracao dos clientes (desemprego recente)."""
    affected = rng.random(len(df)) < UNEMPLOYMENT_SHOCK_FRACTION
    n_affected = int(affected.sum())
    df.loc[affected, "employment_years"] = rng.uniform(
        *UNEMPLOYMENT_SHOCK_YEARS_RANGE, size=n_affected
    ).round(2)
    return df


def simulate_production_dataset(
    reference_df: pd.DataFrame,
    sample_size: int | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Gera o dataset de producao a partir do de referencia, com drift intencional.

    Faz um resample (com reposicao) do dataset de referencia para simular uma
    nova leva de clientes, e aplica a erosao de renda + choque de desemprego
    sobre essa amostra.
    """
    rng = np.random.default_rng(random_state)
    sample_size = sample_size or len(reference_df)

    production_df = reference_df.sample(
        n=sample_size, replace=True, random_state=random_state
    ).reset_index(drop=True)

    # client_id sintetico e unico, deixando claro que e um lote novo (producao).
    production_df["client_id"] = np.arange(1, len(production_df) + 1)

    production_df = _apply_income_erosion(production_df, rng)
    production_df = _apply_unemployment_shock(production_df, rng)

    logger.info(
        "Dataset de producao gerado: %d linhas. "
        "Renda media: referencia=%.0f -> producao=%.0f | "
        "Tempo medio de emprego: referencia=%.1f -> producao=%.1f",
        len(production_df),
        reference_df["income"].mean(),
        production_df["income"].mean(),
        reference_df["employment_years"].mean(),
        production_df["employment_years"].mean(),
    )
    return production_df


def save_production_dataset(df: pd.DataFrame, out_path=PRODUCTION_FILE) -> None:
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Dataset de producao salvo em %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reference_dataset = load_reference_dataset()
    production_dataset = simulate_production_dataset(reference_dataset)
    save_production_dataset(production_dataset)
