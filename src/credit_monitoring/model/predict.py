"""Gera predicoes do modelo baseline para o Dataset de Producao e compara a
distribuicao dos scores com o Dataset de Referencia.

Em producao real, o rotulo verdadeiro (inadimplencia) so e conhecido meses
depois (delayed ground truth) — por isso o monitoramento aqui NAO depende de
acuracia, e sim da distribuicao dos SCORES previstos (e das features de
entrada, cuja mudanca estatistica e formalizada na Etapa 2.3 com o Evidently).

Uso:
    poetry run python -m credit_monitoring.model.predict
"""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from credit_monitoring.model.features import FEATURE_COLUMNS, load_reference_dataset
from credit_monitoring.utils.paths import MODEL_FILE, PRODUCTION_FILE, REPORTS_DIR

logger = logging.getLogger(__name__)

SCORE_COMPARISON_FILE = REPORTS_DIR / "score_comparison.json"
PRODUCTION_PREDICTIONS_FILE = PRODUCTION_FILE.parent / "credit_production_predictions.csv"
HIGH_RISK_THRESHOLD = 0.5


def load_trained_pipeline(path=MODEL_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nao encontrado. Rode "
            "'poetry run python -m credit_monitoring.model.train' antes."
        )
    return joblib.load(path)


def load_production_dataset(path=PRODUCTION_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nao encontrado. Rode "
            "'poetry run python -m credit_monitoring.drift.simulate' antes."
        )
    return pd.read_csv(path)


def generate_predictions(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Gera o score (probabilidade de inadimplencia) para cada linha de `df`."""
    X = df[FEATURE_COLUMNS]
    proba = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "client_id": df["client_id"].values,
            "predicted_proba": proba,
            "predicted_label": (proba >= HIGH_RISK_THRESHOLD).astype(int),
        }
    )


def summarize_scores(predictions: pd.DataFrame) -> dict:
    return {
        "n_samples": int(len(predictions)),
        "mean_score": float(predictions["predicted_proba"].mean()),
        "std_score": float(predictions["predicted_proba"].std()),
        "pct_high_risk": float(predictions["predicted_label"].mean() * 100),
    }


def compare_reference_and_production_scores(
    pipeline, reference_df: pd.DataFrame, production_df: pd.DataFrame
) -> dict:
    """Compara a distribuicao dos scores do modelo entre referencia e producao."""
    reference_summary = summarize_scores(generate_predictions(pipeline, reference_df))
    production_summary = summarize_scores(generate_predictions(pipeline, production_df))

    comparison = {
        "reference": reference_summary,
        "production": production_summary,
        "mean_score_delta": production_summary["mean_score"] - reference_summary["mean_score"],
        "pct_high_risk_delta": (
            production_summary["pct_high_risk"] - reference_summary["pct_high_risk"]
        ),
    }
    logger.info(
        "Score medio: referencia=%.4f -> producao=%.4f (delta=%.4f) | "
        "%% alto risco: referencia=%.1f%% -> producao=%.1f%% (delta=%.1f pp)",
        reference_summary["mean_score"],
        production_summary["mean_score"],
        comparison["mean_score_delta"],
        reference_summary["pct_high_risk"],
        production_summary["pct_high_risk"],
        comparison["pct_high_risk_delta"],
    )
    return comparison


def run_predictions() -> dict:
    """Pipeline de scoring: carrega modelo + datasets, compara e salva os resultados."""
    pipeline = load_trained_pipeline()
    reference_df = load_reference_dataset()
    production_df = load_production_dataset()

    comparison = compare_reference_and_production_scores(pipeline, reference_df, production_df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_COMPARISON_FILE.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    logger.info("Comparacao de scores salva em %s", SCORE_COMPARISON_FILE)

    production_predictions = generate_predictions(pipeline, production_df)
    production_predictions.to_csv(PRODUCTION_PREDICTIONS_FILE, index=False)
    logger.info("Predicoes de producao salvas em %s", PRODUCTION_PREDICTIONS_FILE)

    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_predictions()
