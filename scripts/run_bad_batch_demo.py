"""Demonstra o contrato de dados (Pandera) bloqueando um lote de dados ruim.

Carrega o dataset de referencia, corrompe deliberadamente uma copia dele
(idade invalida, renda nula, duplicata de cliente, valor monetario negativo)
e valida ambos os lotes contra o contrato definido em
`credit_monitoring/data/contracts.py`.

O lote original deve passar; o lote corrompido deve ser BLOQUEADO, com o
relatorio de falhas salvo em `reports/bad_batch_validation_report.csv`.

Uso:
    poetry run python scripts/run_bad_batch_demo.py
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from credit_monitoring.data.contracts import validate_dataset, validation_report
from credit_monitoring.model.features import load_reference_dataset
from credit_monitoring.utils.paths import REPORTS_DIR

logger = logging.getLogger(__name__)

BAD_BATCH_REPORT_FILE = REPORTS_DIR / "bad_batch_validation_report.csv"


def build_bad_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Corrompe deliberadamente uma copia do dataset para forcar o bloqueio."""
    if len(df) < 5:
        raise ValueError("Dataset de referencia tem menos de 5 linhas; nao da para demonstrar.")

    bad_df = df.copy()
    idx = bad_df.index[:5]

    bad_df.loc[idx[0], "age"] = 15.0  # menor de idade (regra: idade > 18)
    bad_df.loc[idx[1], "income"] = None  # renda nula (regra: renda nao pode ser nula)
    bad_df.loc[idx[2], "client_id"] = bad_df.loc[idx[3], "client_id"]  # duplicata de cliente
    bad_df.loc[idx[4], "credit_amount"] = -500.0  # valor monetario invalido (regra: > 0)
    return bad_df


def run_demo() -> bool:
    """Roda a demo de ponta a ponta. Retorna True se o bloqueio funcionou como esperado."""
    reference_df = load_reference_dataset()

    logger.info("Validando o dataset de referencia (esperado: aprovado)...")
    validate_dataset(reference_df)
    logger.info("Dataset de referencia passou no contrato normalmente.")

    logger.info("Gerando lote corrompido propositalmente...")
    bad_batch = build_bad_batch(reference_df)

    failure_cases = validation_report(bad_batch)
    if failure_cases is None:
        logger.error(
            "FALHA NA DEMO: o lote corrompido deveria ter sido bloqueado, mas passou no contrato."
        )
        return False

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    failure_cases.to_csv(BAD_BATCH_REPORT_FILE, index=False)

    violated_columns = sorted(failure_cases["column"].dropna().unique().tolist())
    logger.warning(
        "INGESTAO BLOQUEADA: o lote corrompido violou %d regra(s) do contrato.",
        len(failure_cases),
    )
    logger.warning("Colunas que violaram o contrato: %s", violated_columns)
    logger.info("Relatorio completo de falhas salvo em %s", BAD_BATCH_REPORT_FILE)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ok = run_demo()
    sys.exit(0 if ok else 1)
