"""Treino do classificador baseline de risco de credito.

Usa o pipeline de features definido em `features.py` (imputacao + encoding)
acoplado a um XGBClassifier, treinado sobre o dataset de REFERENCIA (Etapa 1).
O pipeline completo (pre-processamento + modelo) e serializado junto, para
que `predict.py` (Etapa 2) nao precise reimplementar o pre-processamento.

Uso:
    poetry run python -m credit_monitoring.model.train
"""

from __future__ import annotations

import json
import logging

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_monitoring.model.features import (
    build_preprocessor,
    load_reference_dataset,
    train_test_split_reference,
)
from credit_monitoring.utils.paths import MODEL_DIR, MODEL_FILE, REPORTS_DIR

logger = logging.getLogger(__name__)

METRICS_FILE = REPORTS_DIR / "baseline_model_metrics.json"
RANDOM_STATE = 42


def build_model_pipeline() -> Pipeline:
    """Pipeline sklearn: pre-processamento (features.py) + XGBClassifier."""
    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    """Calcula metricas de classificacao no conjunto de teste."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test_samples": int(len(y_test)),
    }
    logger.info(
        "Relatorio de classificacao:\n%s", classification_report(y_test, y_pred)
    )
    return metrics


def train_baseline_model():
    """Treina o pipeline completo, avalia, serializa e salva metricas."""
    df = load_reference_dataset()
    X_train, X_test, y_train, y_test = train_test_split_reference(df)

    pipeline = build_model_pipeline()
    logger.info(
        "Treinando pipeline (preprocessor + XGBClassifier) com %d amostras...",
        len(X_train),
    )
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    logger.info("Modelo salvo em %s", MODEL_FILE)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info("Metricas salvas em %s", METRICS_FILE)

    return pipeline, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_baseline_model()
