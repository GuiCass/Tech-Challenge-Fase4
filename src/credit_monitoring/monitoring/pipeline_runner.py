"""Orquestra as etapas do pipeline com rastreamento de experimentos no MLflow.

Cada etapa (validacao, treino do modelo, predicao, deteccao de drift) roda
registrando parametros, metricas e artefatos relevantes num MLflow run. Em
paralelo, cada etapa tambem e registrada pelo `PipelineEventLogger` (Etapa
3.1) — sucesso/falha e duracao — e pelo `PipelineMetrics` (Etapa 3.3) —
contadores/latencia/volume expostos no formato Prometheus. Os tres
mecanismos juntos dao visibilidade completa do pipeline (Etapa 3:
Observabilidade de Pipelines e Modelos).

Uso:
    poetry run python -m credit_monitoring.monitoring.pipeline_runner
"""

from __future__ import annotations

import logging
import time

import mlflow
import pandas as pd

from credit_monitoring.data.contracts import validate_dataset
from credit_monitoring.drift.detect import (
    DRIFT_REPORT_HTML_FILE,
    DRIFT_SUMMARY_MARKDOWN_FILE,
    run_drift_detection,
)
from credit_monitoring.drift.simulate import save_production_dataset, simulate_production_dataset
from credit_monitoring.model.features import load_reference_dataset
from credit_monitoring.model.predict import SCORE_COMPARISON_FILE, run_predictions
from credit_monitoring.model.train import METRICS_FILE, MODEL_FILE, train_baseline_model
from credit_monitoring.monitoring.logging_config import PipelineEventLogger, configure_logging
from credit_monitoring.monitoring.metrics import PipelineMetrics
from credit_monitoring.utils.paths import PROJECT_ROOT, PRODUCTION_FILE

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT_NAME = "credit-scoring-monitoring"
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").resolve().as_uri()


def configure_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def ensure_production_dataset(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Reusa `data/production/credit_production.csv` se existir, senao simula e salva."""
    if PRODUCTION_FILE.exists():
        return pd.read_csv(PRODUCTION_FILE)

    production_df = simulate_production_dataset(reference_df)
    save_production_dataset(production_df)
    return production_df


def run_validation_stage(events: PipelineEventLogger, metrics: PipelineMetrics | None = None) -> dict:
    """Valida referencia e producao contra o contrato de dados (Etapa 1) antes de seguir."""
    metrics = metrics or PipelineMetrics()
    start = time.perf_counter()
    try:
        with events.stage("validacao_dados"):
            reference_df = load_reference_dataset()
            production_df = ensure_production_dataset(reference_df)
            validate_dataset(reference_df)
            validate_dataset(production_df)
    except Exception:
        metrics.record_stage("validacao_dados", "failure", time.perf_counter() - start)
        raise

    result = {"reference_rows": len(reference_df), "production_rows": len(production_df)}
    metrics.record_stage(
        "validacao_dados",
        "success",
        time.perf_counter() - start,
        rows=result["reference_rows"] + result["production_rows"],
    )
    return result


def run_training_stage(events: PipelineEventLogger, metrics: PipelineMetrics | None = None) -> dict:
    """Treina o modelo baseline dentro de um MLflow run (params + metricas + artifacts)."""
    metrics = metrics or PipelineMetrics()
    start = time.perf_counter()
    try:
        with events.stage("treino_modelo"):
            with mlflow.start_run(run_name="train_baseline_model"):
                pipeline, model_metrics = train_baseline_model()

                classifier = pipeline.named_steps["classifier"]
                mlflow.log_params(
                    {
                        "model_type": type(classifier).__name__,
                        "n_estimators": classifier.n_estimators,
                        "max_depth": classifier.max_depth,
                        "learning_rate": classifier.learning_rate,
                    }
                )
                mlflow.log_metrics(
                    {k: v for k, v in model_metrics.items() if isinstance(v, (int, float))}
                )
                mlflow.log_artifact(str(MODEL_FILE))
                mlflow.log_artifact(str(METRICS_FILE))
    except Exception:
        metrics.record_stage("treino_modelo", "failure", time.perf_counter() - start)
        raise

    metrics.record_stage(
        "treino_modelo",
        "success",
        time.perf_counter() - start,
        rows=model_metrics.get("n_test_samples"),
    )
    return model_metrics


def run_prediction_stage(events: PipelineEventLogger, metrics: PipelineMetrics | None = None) -> dict:
    """Gera e compara scores (referencia x producao) dentro de um MLflow run."""
    metrics = metrics or PipelineMetrics()
    start = time.perf_counter()
    try:
        with events.stage("predicao"):
            with mlflow.start_run(run_name="score_production"):
                comparison = run_predictions()

                mlflow.log_metrics(
                    {
                        "production_mean_score": comparison["production"]["mean_score"],
                        "production_pct_high_risk": comparison["production"]["pct_high_risk"],
                        "mean_score_delta": comparison["mean_score_delta"],
                    }
                )
                mlflow.log_artifact(str(SCORE_COMPARISON_FILE))
    except Exception:
        metrics.record_stage("predicao", "failure", time.perf_counter() - start)
        raise

    metrics.record_stage(
        "predicao",
        "success",
        time.perf_counter() - start,
        rows=comparison["production"]["n_samples"],
    )
    return comparison


def run_drift_stage(events: PipelineEventLogger, metrics: PipelineMetrics | None = None) -> dict:
    """Roda a deteccao de drift dentro de um MLflow run, logando os relatorios como artifacts."""
    metrics = metrics or PipelineMetrics()
    start = time.perf_counter()
    try:
        with events.stage("deteccao_drift"):
            with mlflow.start_run(run_name="drift_detection"):
                reference_df = load_reference_dataset()
                production_df = ensure_production_dataset(reference_df)

                summary = run_drift_detection(reference_df, production_df)

                mlflow.log_metrics(
                    {
                        "number_of_drifted_columns": summary["number_of_drifted_columns"],
                        "share_of_drifted_columns": summary["share_of_drifted_columns"],
                    }
                )
                mlflow.log_param("dataset_drift_detected", summary["dataset_drift_detected"])
                mlflow.log_artifact(str(DRIFT_REPORT_HTML_FILE))
                mlflow.log_artifact(str(DRIFT_SUMMARY_MARKDOWN_FILE))
    except Exception:
        metrics.record_stage("deteccao_drift", "failure", time.perf_counter() - start)
        raise

    metrics.record_stage(
        "deteccao_drift",
        "success",
        time.perf_counter() - start,
        rows=len(production_df),
    )
    if summary["dataset_drift_detected"]:
        metrics.record_drift_alert()
    return summary


def run_full_pipeline() -> dict:
    """Roda o pipeline fim-a-fim: validacao -> treino -> predicao -> drift.

    Usado pelo `scripts/run_pipeline.py` (Etapa 3.4) para alimentar o
    dashboard HTML consolidado com o resultado de cada etapa.
    """
    configure_logging()
    configure_mlflow()
    events = PipelineEventLogger()
    metrics = PipelineMetrics()

    run_validation_stage(events, metrics)
    training_metrics = run_training_stage(events, metrics)
    prediction_comparison = run_prediction_stage(events, metrics)
    drift_summary = run_drift_stage(events, metrics)

    metrics.write_textfile()

    return {
        "training_metrics": training_metrics,
        "prediction_comparison": prediction_comparison,
        "drift_summary": drift_summary,
        "events": events.read_events(),
    }


if __name__ == "__main__":
    run_full_pipeline()
