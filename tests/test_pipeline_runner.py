import mlflow
import pytest

import credit_monitoring.monitoring.pipeline_runner as runner_module
from credit_monitoring.monitoring.logging_config import PipelineEventLogger


@pytest.fixture(autouse=True)
def isolated_mlflow_tracking(tmp_path):
    """Aponta o MLflow para um store temporario, isolado do mlruns/ real do projeto."""
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test-experiment")
    yield
    mlflow.end_run()


def test_run_training_stage_logs_params_metrics_and_artifacts(
    monkeypatch, synthetic_reference_df, tmp_path
):
    model_file = tmp_path / "model.joblib"
    metrics_file = tmp_path / "metrics.json"
    model_file.write_bytes(b"fake-model-bytes")
    metrics_file.write_text("{}")

    from credit_monitoring.model.train import build_model_pipeline, evaluate
    from credit_monitoring.model.features import train_test_split_reference

    def fake_train_baseline_model():
        X_train, X_test, y_train, y_test = train_test_split_reference(synthetic_reference_df)
        pipeline = build_model_pipeline()
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        return pipeline, metrics

    monkeypatch.setattr(runner_module, "train_baseline_model", fake_train_baseline_model)
    monkeypatch.setattr(runner_module, "MODEL_FILE", model_file)
    monkeypatch.setattr(runner_module, "METRICS_FILE", metrics_file)

    events = PipelineEventLogger(events_file=tmp_path / "events.jsonl")
    metrics = runner_module.run_training_stage(events)

    assert "accuracy" in metrics
    recorded = events.read_events()
    assert recorded[0]["stage"] == "treino_modelo"
    assert recorded[0]["status"] == "success"

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test-experiment")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params["model_type"] == "XGBClassifier"
    assert "accuracy" in runs[0].data.metrics


def test_run_drift_stage_logs_metrics_and_artifacts(
    monkeypatch, synthetic_reference_df, tmp_path
):
    import credit_monitoring.drift.detect as detect_module
    from credit_monitoring.drift.simulate import simulate_production_dataset

    html_file = tmp_path / "drift_report.html"
    summary_file = tmp_path / "drift_summary.json"
    markdown_file = tmp_path / "drift_summary.md"

    production_df = simulate_production_dataset(synthetic_reference_df, random_state=123)

    monkeypatch.setattr(runner_module, "load_reference_dataset", lambda: synthetic_reference_df)
    monkeypatch.setattr(runner_module, "PRODUCTION_FILE", tmp_path / "does_not_exist.csv")
    monkeypatch.setattr(
        runner_module, "simulate_production_dataset", lambda df: production_df
    )
    # evita que `ensure_production_dataset` grave o CSV simulado no `data/production/`
    # real do projeto quando o arquivo "nao existe" (PRODUCTION_FILE patcheado acima).
    monkeypatch.setattr(runner_module, "save_production_dataset", lambda df: None)

    # `run_drift_detection` (chamado de dentro de `run_drift_stage`) escreve nos
    # caminhos do PROPRIO modulo `detect`, nao nos que o `pipeline_runner` importou
    # por nome — por isso os dois precisam ser isolados para o mesmo tmp_path.
    monkeypatch.setattr(detect_module, "DRIFT_REPORTS_DIR", tmp_path)
    monkeypatch.setattr(detect_module, "DRIFT_REPORT_HTML_FILE", html_file)
    monkeypatch.setattr(detect_module, "DRIFT_SUMMARY_FILE", summary_file)
    monkeypatch.setattr(detect_module, "DRIFT_SUMMARY_MARKDOWN_FILE", markdown_file)
    monkeypatch.setattr(runner_module, "DRIFT_REPORT_HTML_FILE", html_file)
    monkeypatch.setattr(runner_module, "DRIFT_SUMMARY_MARKDOWN_FILE", markdown_file)

    events = PipelineEventLogger(events_file=tmp_path / "events.jsonl")
    summary = runner_module.run_drift_stage(events)

    assert html_file.exists()
    assert markdown_file.exists()

    assert "drifted_columns" in summary
    recorded = events.read_events()
    assert recorded[0]["stage"] == "deteccao_drift"
    assert recorded[0]["status"] == "success"

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test-experiment")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert "number_of_drifted_columns" in runs[0].data.metrics


def test_run_training_stage_records_failure_event_on_exception(monkeypatch, tmp_path):
    def failing_train():
        raise RuntimeError("falha proposital no treino")

    monkeypatch.setattr(runner_module, "train_baseline_model", failing_train)

    events = PipelineEventLogger(events_file=tmp_path / "events.jsonl")
    with pytest.raises(RuntimeError, match="falha proposital"):
        runner_module.run_training_stage(events)

    recorded = events.read_events()
    assert recorded[0]["status"] == "failure"


def test_run_validation_stage_passes_for_valid_datasets(
    monkeypatch, synthetic_reference_df, tmp_path
):
    from credit_monitoring.drift.simulate import simulate_production_dataset
    from credit_monitoring.monitoring.metrics import PipelineMetrics

    production_df = simulate_production_dataset(synthetic_reference_df, random_state=123)

    monkeypatch.setattr(runner_module, "load_reference_dataset", lambda: synthetic_reference_df)
    monkeypatch.setattr(runner_module, "PRODUCTION_FILE", tmp_path / "does_not_exist.csv")
    monkeypatch.setattr(runner_module, "simulate_production_dataset", lambda df: production_df)
    monkeypatch.setattr(runner_module, "save_production_dataset", lambda df: None)

    events = PipelineEventLogger(events_file=tmp_path / "events.jsonl")
    metrics = PipelineMetrics()

    result = runner_module.run_validation_stage(events, metrics)

    assert result == {
        "reference_rows": len(synthetic_reference_df),
        "production_rows": len(production_df),
    }
    recorded = events.read_events()
    assert recorded[0]["stage"] == "validacao_dados"
    assert recorded[0]["status"] == "success"


def test_run_validation_stage_blocks_and_records_failure_for_invalid_dataset(
    monkeypatch, synthetic_reference_df, tmp_path
):
    invalid_df = synthetic_reference_df.copy()
    invalid_df.loc[0, "age"] = 10  # viola a regra de idade minima do contrato

    monkeypatch.setattr(runner_module, "load_reference_dataset", lambda: invalid_df)
    monkeypatch.setattr(runner_module, "PRODUCTION_FILE", tmp_path / "does_not_exist.csv")
    monkeypatch.setattr(runner_module, "simulate_production_dataset", lambda df: df)
    monkeypatch.setattr(runner_module, "save_production_dataset", lambda df: None)

    events = PipelineEventLogger(events_file=tmp_path / "events.jsonl")

    with pytest.raises(Exception):
        runner_module.run_validation_stage(events)

    recorded = events.read_events()
    assert recorded[0]["stage"] == "validacao_dados"
    assert recorded[0]["status"] == "failure"


def test_run_prediction_stage_logs_metrics_and_artifact(
    monkeypatch, synthetic_reference_df, tmp_path
):
    from credit_monitoring.drift.simulate import simulate_production_dataset

    production_df = simulate_production_dataset(synthetic_reference_df, random_state=123)
    score_comparison_file = tmp_path / "score_comparison.json"
    score_comparison_file.write_text("{}")

    def fake_run_predictions():
        from credit_monitoring.model.train import build_model_pipeline
        from credit_monitoring.model.features import train_test_split_reference
        from credit_monitoring.model.predict import compare_reference_and_production_scores

        X_train, X_test, y_train, y_test = train_test_split_reference(synthetic_reference_df)
        pipeline = build_model_pipeline()
        pipeline.fit(X_train, y_train)
        return compare_reference_and_production_scores(
            pipeline, synthetic_reference_df, production_df
        )

    monkeypatch.setattr(runner_module, "run_predictions", fake_run_predictions)
    monkeypatch.setattr(runner_module, "SCORE_COMPARISON_FILE", score_comparison_file)

    events = PipelineEventLogger(events_file=tmp_path / "events.jsonl")
    comparison = runner_module.run_prediction_stage(events)

    assert "mean_score_delta" in comparison
    recorded = events.read_events()
    assert recorded[0]["stage"] == "predicao"
    assert recorded[0]["status"] == "success"

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test-experiment")
    runs = client.search_runs([experiment.experiment_id])
    assert any("production_mean_score" in run.data.metrics for run in runs)


def test_run_full_pipeline_orchestrates_all_stages_and_writes_metrics(monkeypatch, tmp_path):
    from credit_monitoring.monitoring.logging_config import PipelineEventLogger as _Events
    from credit_monitoring.monitoring.metrics import PipelineMetrics as _Metrics

    call_order = []

    def fake_validation(events, metrics):
        call_order.append("validacao_dados")
        return {"reference_rows": 1, "production_rows": 1}

    def fake_training(events, metrics):
        call_order.append("treino_modelo")
        return {"accuracy": 0.9, "n_test_samples": 10}

    def fake_prediction(events, metrics):
        call_order.append("predicao")
        return {"mean_score_delta": 0.05}

    def fake_drift(events, metrics):
        call_order.append("deteccao_drift")
        return {"dataset_drift_detected": True}

    monkeypatch.setattr(runner_module, "configure_logging", lambda: None)
    monkeypatch.setattr(runner_module, "configure_mlflow", lambda: None)
    monkeypatch.setattr(runner_module, "run_validation_stage", fake_validation)
    monkeypatch.setattr(runner_module, "run_training_stage", fake_training)
    monkeypatch.setattr(runner_module, "run_prediction_stage", fake_prediction)
    monkeypatch.setattr(runner_module, "run_drift_stage", fake_drift)

    written = {}
    monkeypatch.setattr(
        _Metrics, "write_textfile", lambda self, *a, **kw: written.setdefault("called", True)
    )
    monkeypatch.setattr(
        _Events, "read_events", lambda self: [{"stage": "treino_modelo", "status": "success"}]
    )

    result = runner_module.run_full_pipeline()

    assert call_order == ["validacao_dados", "treino_modelo", "predicao", "deteccao_drift"]
    assert written.get("called") is True
    assert result["training_metrics"]["accuracy"] == 0.9
    assert result["prediction_comparison"]["mean_score_delta"] == 0.05
    assert result["drift_summary"]["dataset_drift_detected"] is True
    assert result["events"] == [{"stage": "treino_modelo", "status": "success"}]
