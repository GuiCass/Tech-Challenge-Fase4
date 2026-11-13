"""Orquestra as etapas do pipeline com rastreamento de experimentos no MLflow.

Cada etapa (treino do modelo, deteccao de drift) roda dentro de um MLflow run,
registrando parametros, metricas e artefatos (modelo serializado, relatorios
de drift). Em paralelo, cada etapa tambem e registrada pelo
`PipelineEventLogger` (Etapa 3.1), com sucesso/falha e duracao — os dois
mecanismos juntos dao visibilidade completa do pipeline (Etapa 3:
Observabilidade de Pipelines e Modelos).

Uso:
    poetry run python -m credit_monitoring.monitoring.pipeline_runner
"""

from __future__ import annotations

import logging

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


def run_training_stage(events: PipelineEventLogger) -> dict:
    """Treina o modelo baseline dentro de um MLflow run (params + metricas + artifacts)."""
    with events.stage("treino_modelo"):
        with mlflow.start_run(run_name="train_baseline_model"):
            pipeline, metrics = train_baseline_model()

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
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            mlflow.log_artifact(str(MODEL_FILE))
            mlflow.log_artifact(str(METRICS_FILE))

    return metrics


def run_drift_stage(events: PipelineEventLogger) -> dict:
    """Roda a deteccao de drift dentro de um MLflow run, logando os relatorios como artifacts."""
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

    return summary


if __name__ == "__main__":
    configure_logging()
    configure_mlflow()
    pipeline_events = PipelineEventLogger()

    run_training_stage(pipeline_events)
    run_drift_stage(pipeline_events)
