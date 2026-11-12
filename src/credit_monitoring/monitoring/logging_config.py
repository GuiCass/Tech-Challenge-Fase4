"""Logging estruturado do pipeline de credito.

Centraliza a configuracao de logging (console + arquivo) e fornece o
`PipelineEventLogger`, que registra cada ETAPA do pipeline (validacao de
dados, treino, predicao, deteccao de drift) como um evento estruturado em
JSON Lines — consumido depois pelo dashboard de observabilidade (Etapa 3.4)
para mostrar sucesso/falha, latencia e volume de cada execucao.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from credit_monitoring.utils.paths import PROJECT_ROOT

LOGS_DIR = PROJECT_ROOT / "logs"
PIPELINE_LOG_FILE = LOGS_DIR / "pipeline.log"
PIPELINE_EVENTS_FILE = LOGS_DIR / "pipeline_events.jsonl"


def configure_logging(log_file=PIPELINE_LOG_FILE, level: int = logging.INFO) -> None:
    """Configura o logger raiz com saida simultanea para console e arquivo."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # evita handlers duplicados se chamado mais de uma vez

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


@dataclass
class PipelineEvent:
    stage: str
    status: str  # "success" | "failure"
    duration_seconds: float
    message: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class PipelineEventLogger:
    """Registra eventos de cada etapa do pipeline em JSON Lines.

    Uso:
        events = PipelineEventLogger()
        with events.stage("validacao_dados", n_rows=len(df)):
            validate_dataset(df)
    """

    def __init__(self, events_file=PIPELINE_EVENTS_FILE):
        self.events_file = events_file
        self.logger = logging.getLogger("credit_monitoring.pipeline")

    def _write_event(self, event: PipelineEvent) -> None:
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")

    @contextmanager
    def stage(self, name: str, **metadata):
        """Context manager: mede a duracao e registra sucesso/falha automaticamente."""
        start = time.perf_counter()
        self.logger.info("Iniciando etapa '%s'...", name)
        try:
            yield
        except Exception as exc:
            duration = time.perf_counter() - start
            self.logger.error("Etapa '%s' FALHOU apos %.2fs: %s", name, duration, exc)
            self._write_event(
                PipelineEvent(
                    stage=name,
                    status="failure",
                    duration_seconds=duration,
                    message=str(exc),
                    metadata=metadata,
                )
            )
            raise
        else:
            duration = time.perf_counter() - start
            self.logger.info("Etapa '%s' concluida com sucesso em %.2fs.", name, duration)
            self._write_event(
                PipelineEvent(
                    stage=name,
                    status="success",
                    duration_seconds=duration,
                    metadata=metadata,
                )
            )

    def read_events(self) -> list[dict]:
        """Le todos os eventos ja registrados (usado pelo dashboard, Etapa 3.4)."""
        if not self.events_file.exists():
            return []
        with open(self.events_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
