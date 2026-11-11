"""Detecta Data Drift entre o Dataset de Referencia e o Dataset de Producao.

Usa o Evidently AI com dois testes estatisticos, conforme pedido no desafio:
- Kolmogorov-Smirnov (KS) para features NUMERICAS.
- PSI (Population Stability Index) para features CATEGORICAS.

Gera:
- `reports/drift/drift_report.html`: relatorio visual completo do Evidently.
- `reports/drift/drift_summary.json`: resumo programatico (quais colunas
  sofreram drift, com qual metrica e score).

Uso:
    poetry run python -m credit_monitoring.drift.detect
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from credit_monitoring.model.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from credit_monitoring.utils.paths import DRIFT_REPORTS_DIR

logger = logging.getLogger(__name__)

DRIFT_REPORT_HTML_FILE = DRIFT_REPORTS_DIR / "drift_report.html"
DRIFT_SUMMARY_FILE = DRIFT_REPORTS_DIR / "drift_summary.json"


def build_column_mapping() -> ColumnMapping:
    return ColumnMapping(
        target=TARGET_COLUMN,
        numerical_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )


def build_drift_report() -> Report:
    """Report do Evidently: KS para numericas, PSI para categoricas."""
    return Report(metrics=[DataDriftPreset(num_stattest="ks", cat_stattest="psi")])


def summarize_drift_result(result: dict) -> dict:
    """Extrai do `Report.as_dict()` quais colunas sofreram drift e com qual metrica."""
    dataset_drift_metric = next(
        m for m in result["metrics"] if m["metric"] == "DatasetDriftMetric"
    )["result"]
    data_drift_table = next(
        m for m in result["metrics"] if m["metric"] == "DataDriftTable"
    )["result"]

    drifted_columns = [
        {
            "column": name,
            "column_type": info["column_type"],
            "stattest_name": info["stattest_name"],
            "drift_score": info["drift_score"],
        }
        for name, info in data_drift_table["drift_by_columns"].items()
        if info["drift_detected"]
    ]
    # ordena pelas colunas mais degradadas primeiro (menor p-valor / maior PSI)
    drifted_columns.sort(key=lambda c: c["drift_score"])

    return {
        "dataset_drift_detected": dataset_drift_metric["dataset_drift"],
        "share_of_drifted_columns": dataset_drift_metric["share_of_drifted_columns"],
        "number_of_drifted_columns": dataset_drift_metric["number_of_drifted_columns"],
        "number_of_columns": dataset_drift_metric["number_of_columns"],
        "drifted_columns": drifted_columns,
    }


def run_drift_detection(
    reference_df: pd.DataFrame, production_df: pd.DataFrame
) -> dict:
    """Roda o Evidently, salva o HTML e retorna o resumo de colunas com drift."""
    report = build_drift_report()
    report.run(
        reference_data=reference_df,
        current_data=production_df,
        column_mapping=build_column_mapping(),
    )

    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(DRIFT_REPORT_HTML_FILE))
    logger.info("Relatorio HTML de drift salvo em %s", DRIFT_REPORT_HTML_FILE)

    summary = summarize_drift_result(report.as_dict())
    DRIFT_SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Resumo de drift salvo em %s", DRIFT_SUMMARY_FILE)

    if summary["drifted_columns"]:
        logger.warning(
            "DRIFT DETECTADO em %d/%d colunas: %s",
            summary["number_of_drifted_columns"],
            summary["number_of_columns"],
            [c["column"] for c in summary["drifted_columns"]],
        )
    else:
        logger.info("Nenhuma coluna com drift detectado.")

    return summary


if __name__ == "__main__":
    from credit_monitoring.drift.simulate import simulate_production_dataset
    from credit_monitoring.model.features import load_reference_dataset
    from credit_monitoring.utils.paths import PRODUCTION_FILE

    logging.basicConfig(level=logging.INFO)

    reference_dataset = load_reference_dataset()
    production_dataset = (
        pd.read_csv(PRODUCTION_FILE)
        if PRODUCTION_FILE.exists()
        else simulate_production_dataset(reference_dataset)
    )
    run_drift_detection(reference_dataset, production_dataset)
