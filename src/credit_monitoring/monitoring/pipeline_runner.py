"""Orquestra as etapas do pipeline com rastreamento de experimentos no MLflow.

Cada etapa (treino do modelo, deteccao de drift) roda dentro de um MLflow run,
registrando parametros, metricas e artefatos (modelo serializado, relatorios
de drift). Em paralelo, cada etapa tambem e registrada pelo
`PipelineEventLogger` (Etapa 3.1) — sucesso/falha e duracao — e pelo
`PipelineMetrics` (Etapa 3.3) — contadores/latencia/volume expostos no
formato Prometheus. Os tres mecanismos juntos dao visibilidade completa do
pipeline (Etapa 3: Observabilidade de Pipelines e Modelos).

Uso:
    poetry run python -m credit_monitoring.monitoring.pipeline_runner
"""

from __future__ import annotations

import logging
import time

import mlflow
import pandas as pd

from credit_monitoring.drift.detect import (
    DRIFT_REPORT_HTML_FILE,
    DRIFT_SUMMARY_MARKDOWN_FILE,
    run_drift_detection,
)
from credit_monitoring.drift.simulate import simulate_production_dataset
from credit_monitoring.model.features import load_reference_dataset
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


def run_drift_stage(events: PipelineEventLogger, metrics: PipelineMetrics | None = None) -> dict:
    """Roda a deteccao de drift dentro de um MLflow run, logando os relatorios como artifacts."""
    metrics = metrics or PipelineMetrics()
    start = time.perf_counter()
    try:
        with events.stage("deteccao_drift"):
            with mlflow.start_run(run_name="drift_detection"):
                reference_df = load_reference_dataset()
                production_df = (
                    pd.read_csv(PRODUCTION_FILE)
                    if PRODUCTION_FILE.exists()
                    else simulate_production_dataset(reference_df)
                )

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


if __name__ == "__main__":
    configure_logging()
    configure_mlflow()
    pipeline_events = PipelineEventLogger()
    pipeline_metrics = PipelineMetrics()

    run_training_stage(pipeline_events, pipeline_metrics)
    run_drift_stage(pipeline_events, pipeline_metrics)
    pipeline_metrics.write_textfile()
