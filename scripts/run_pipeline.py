"""CLI que roda o pipeline completo de observabilidade do modelo de credito.

Etapas (nesta ordem): validacao de dados -> treino do modelo -> predicao/
scoring em producao -> deteccao de drift. Cada etapa fica registrada no log
estruturado (JSON Lines), num MLflow run e nas metricas Prometheus
(`metrics/pipeline_metrics.prom`). Ao final, monta o dashboard HTML
consolidado em `reports/dashboard/dashboard.html`.

Uso:
    poetry run python scripts/run_pipeline.py
"""

from __future__ import annotations

import logging

from credit_monitoring.monitoring.dashboard import DASHBOARD_HTML_FILE, save_dashboard
from credit_monitoring.monitoring.pipeline_runner import run_full_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    result = run_full_pipeline()
    save_dashboard(result)
    logger.info("Dashboard consolidado salvo em %s", DASHBOARD_HTML_FILE)


if __name__ == "__main__":
    main()
