"""Metricas Prometheus do pipeline de credito.

O pipeline roda em lote (nao e um servico de longa duracao), entao as
metricas seguem o padrao "textfile collector" do Prometheus: sao escritas em
um arquivo `.prom` que um node_exporter configurado com
`--collector.textfile.directory` (ou um `pushgateway`) consegue raspar entre
uma execucao e outra.

Uso:
    metrics = PipelineMetrics()
    metrics.record_stage("treino_modelo", "success", duration_seconds=12.3, rows=20000)
    metrics.record_drift_alert()
    metrics.write_textfile()
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, write_to_textfile

from credit_monitoring.utils.paths import PROJECT_ROOT

METRICS_DIR = PROJECT_ROOT / "metrics"
PROMETHEUS_TEXTFILE = METRICS_DIR / "pipeline_metrics.prom"


class PipelineMetrics:
    """Agrupa as metricas Prometheus de uma execucao do pipeline."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()

        self.runs_total = Counter(
            "pipeline_stage_runs_total",
            "Numero de execucoes de cada etapa do pipeline, por status",
            ["stage", "status"],
            registry=self.registry,
        )
        self.stage_duration_seconds = Histogram(
            "pipeline_stage_duration_seconds",
            "Duracao (em segundos) de cada etapa do pipeline",
            ["stage"],
            registry=self.registry,
        )
        self.rows_processed = Gauge(
            "pipeline_rows_processed",
            "Numero de linhas processadas na ultima execucao de cada etapa",
            ["stage"],
            registry=self.registry,
        )
        self.last_run_timestamp = Gauge(
            "pipeline_last_run_timestamp_seconds",
            "Timestamp Unix da ultima execucao de cada etapa, por status",
            ["stage", "status"],
            registry=self.registry,
        )
        self.drift_alerts_total = Counter(
            "pipeline_drift_alerts_total",
            "Numero de execucoes em que drift foi detectado no dataset de producao",
            registry=self.registry,
        )

    def record_stage(
        self,
        stage: str,
        status: str,
        duration_seconds: float,
        rows: int | None = None,
    ) -> None:
        """Registra o resultado (sucesso/falha), a latencia e o volume de uma etapa."""
        self.runs_total.labels(stage=stage, status=status).inc()
        self.stage_duration_seconds.labels(stage=stage).observe(duration_seconds)
        self.last_run_timestamp.labels(stage=stage, status=status).set_to_current_time()
        if rows is not None:
            self.rows_processed.labels(stage=stage).set(rows)

    def record_drift_alert(self) -> None:
        """Incrementa o contador de alertas de drift (dataset_drift_detected=True)."""
        self.drift_alerts_total.inc()

    def write_textfile(self, path=PROMETHEUS_TEXTFILE) -> None:
        """Escreve as metricas no formato texto do Prometheus (textfile collector)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        write_to_textfile(str(path), self.registry)
