import logging

import pytest

from credit_monitoring.monitoring.logging_config import PipelineEventLogger, configure_logging


def test_configure_logging_writes_to_file(tmp_path):
    log_file = tmp_path / "pipeline.log"
    configure_logging(log_file=log_file)

    logging.getLogger("test").info("mensagem de teste")

    assert log_file.exists()
    assert "mensagem de teste" in log_file.read_text(encoding="utf-8")


def test_stage_records_success_event(tmp_path):
    events_file = tmp_path / "events.jsonl"
    events = PipelineEventLogger(events_file=events_file)

    with events.stage("validacao_dados", n_rows=100):
        pass

    recorded = events.read_events()
    assert len(recorded) == 1
    assert recorded[0]["stage"] == "validacao_dados"
    assert recorded[0]["status"] == "success"
    assert recorded[0]["metadata"] == {"n_rows": 100}
    assert recorded[0]["duration_seconds"] >= 0


def test_stage_records_failure_event_and_reraises(tmp_path):
    events_file = tmp_path / "events.jsonl"
    events = PipelineEventLogger(events_file=events_file)

    with pytest.raises(ValueError, match="erro proposital"):
        with events.stage("treino_modelo"):
            raise ValueError("erro proposital")

    recorded = events.read_events()
    assert len(recorded) == 1
    assert recorded[0]["status"] == "failure"
    assert "erro proposital" in recorded[0]["message"]


def test_read_events_returns_empty_list_when_no_file(tmp_path):
    events = PipelineEventLogger(events_file=tmp_path / "does_not_exist.jsonl")
    assert events.read_events() == []


def test_multiple_stages_appended_in_order(tmp_path):
    events_file = tmp_path / "events.jsonl"
    events = PipelineEventLogger(events_file=events_file)

    with events.stage("etapa_1"):
        pass
    with pytest.raises(RuntimeError):
        with events.stage("etapa_2"):
            raise RuntimeError("falhou")
    with events.stage("etapa_3"):
        pass

    recorded = events.read_events()
    assert [e["stage"] for e in recorded] == ["etapa_1", "etapa_2", "etapa_3"]
    assert [e["status"] for e in recorded] == ["success", "failure", "success"]
