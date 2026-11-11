"""Contrato de qualidade de dados (Data Contract) do dataset de credito.

Define, via Pandera, as regras que qualquer lote de dados (referencia OU
producao) precisa satisfazer antes de alimentar o modelo. Regras rigidas
obrigatorias do desafio:

1. Sem nulos em colunas criticas (id, idade, renda, target).
2. Range de idade (maior de 18 anos, abaixo de 100).
3. Limites de renda e de valores monetarios (> 0).
4. Sem duplicatas de cliente (`client_id` unico).

Alem dessas, o schema tambem valida tipos, enum de categorias e ranges de
outras colunas, para tornar o contrato mais robusto.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

CREDIT_DATA_SCHEMA = DataFrameSchema(
    columns={
        "client_id": Column(pa.Int64, unique=True, nullable=False),
        "target": Column(pa.Int64, Check.isin([0, 1]), nullable=False),
        "gender": Column(str, Check.isin(["M", "F", "XNA"]), nullable=True),
        "num_children": Column(pa.Int64, Check.greater_than_or_equal_to(0), nullable=False),
        "income": Column(float, Check.greater_than(0), nullable=False),
        "credit_amount": Column(float, Check.greater_than(0), nullable=False),
        "annuity_amount": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "education": Column(str, nullable=True),
        "family_status": Column(str, nullable=True),
        "housing_type": Column(str, nullable=True),
        # Regra obrigatoria: idade estritamente maior que 18 anos.
        "age": Column(float, Check.in_range(18, 100, include_min=False), nullable=False),
        "employment_years": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "owns_car": Column(str, Check.isin(["Y", "N"]), nullable=True),
        "owns_realty": Column(str, Check.isin(["Y", "N"]), nullable=True),
        "occupation_type": Column(str, nullable=True),
        "region_rating": Column(pa.Int64, Check.isin([1, 2, 3]), nullable=False),
        "organization_type": Column(str, nullable=True),
    },
    coerce=True,
    strict=False,
)


def validate_dataset(df: pd.DataFrame, schema: DataFrameSchema = CREDIT_DATA_SCHEMA) -> pd.DataFrame:
    """Valida `df` contra o contrato, levantando todos os erros encontrados.

    Usa `lazy=True` para acumular TODAS as violacoes (nao so a primeira),
    o que e essencial para o relatorio de bloqueio de ingestao (Etapa 1).

    Retorna o proprio dataframe (coagido aos tipos do schema) quando valido.
    Levanta `pandera.errors.SchemaErrors` quando invalido.
    """
    return schema.validate(df, lazy=True)


def validation_report(df: pd.DataFrame, schema: DataFrameSchema = CREDIT_DATA_SCHEMA) -> pd.DataFrame | None:
    """Retorna o dataframe de falhas do Pandera (ou None se o lote for valido).

    Util para logar/serializar os motivos do bloqueio sem interromper o fluxo.
    """
    try:
        schema.validate(df, lazy=True)
        return None
    except SchemaErrors as exc:
        return exc.failure_cases
