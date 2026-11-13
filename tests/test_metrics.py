from prometheus_client import CollectorRegistry

from credit_monitoring.monitoring.metrics import PipelineMetrics


def _sample_value(registry: CollectorRegistry, name: str, labels: dict | None = None):
    labels = labels or {}
    for family in registry.collect():
        for sample in family.samples:
            if sample.name == name and all(
                sample.labels.get(k) == v for k, v in labels.items()
            ):
                return sample.value
    return None


def test_record_stage_success_updates_counters_duration_and_rows():
    metrics = PipelineMetrics(registry=CollectorRegistry())

    metrics.record_stage("treino_modelo", "success", duration_seconds=1.5, rows=1000)

    assert (
        _sample_value(
            metrics.registry,
            "pipeline_stage_runs_total",
            {"stage": "treino_modelo", "status": "success"},
        )
        == 1.0
    )
    assert (
        _sample_value(
            metrics.registry, "pipeline_rows_processed", {"stage": "treino_modelo"}
        )
        == 1000.0
    )
    assert (
        _sample_value(
            metrics.registry,
            "pipeline_stage_duration_seconds_sum",
            {"stage": "treino_modelo"},
        )
        == 1.5
    )
    assert (
        _sample_value(
            metrics.registry,
            "pipeline_last_run_timestamp_seconds",
            {"stage": "treino_modelo", "status": "success"},
        )
        > 0
    )


def test_record_stage_without_rows_leaves_rows_gauge_unset():
    metrics = PipelineMetrics(registry=CollectorRegistry())

    metrics.record_stage("deteccao_drift", "failure", duration_seconds=0.2)

    assert (
        _sample_value(
            metrics.registry,
            "pipeline_stage_runs_total",
            {"stage": "deteccao_drift", "status": "failure"},
        )
        == 1.0
    )
    assert (
        _sample_value(
            metrics.registry, "pipeline_rows_processed", {"stage": "deteccao_drift"}
        )
        is None
    )


def test_record_stage_accumulates_across_multiple_calls():
    metrics = PipelineMetrics(registry=CollectorRegistry())

    metrics.record_stage("treino_modelo", "success", duration_seconds=1.0, rows=100)
    metrics.record_stage("treino_modelo", "success", duration_seconds=2.0, rows=200)

    assert (
        _sample_value(
            metrics.registry,
            "pipeline_stage_runs_total",
            {"stage": "treino_modelo", "status": "success"},
        )
        == 2.0
    )
    assert (
        _sample_value(
            metrics.registry, "pipeline_rows_processed", {"stage": "treino_modelo"}
        )
        == 200.0
    )


def test_record_drift_alert_increments_counter():
    metrics = PipelineMetrics(registry=CollectorRegistry())

    metrics.record_drift_alert()
    metrics.record_drift_alert()

    assert _sample_value(metrics.registry, "pipeline_drift_alerts_total") == 2.0


def test_write_textfile_produces_prometheus_text_format(tmp_path):
    metrics = PipelineMetrics(registry=CollectorRegistry())
    metrics.record_stage("treino_modelo", "success", duration_seconds=1.5, rows=1000)
    metrics.record_drift_alert()

    out_file = tmp_path / "pipeline_metrics.prom"
    metrics.write_textfile(out_file)

    content = out_file.read_text(encoding="utf-8")
    assert "pipeline_stage_runs_total" in content
    assert "pipeline_rows_processed" in content
    assert "pipeline_drift_alerts_total" in content
