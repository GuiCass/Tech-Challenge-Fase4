# Tech Challenge — Fase 4: Observabilidade, Drift e Governança de Modelo de Crédito

> Pós Tech (FIAP) — Tema Central: Observabilidade Contínua, Detecção de Drift,
> Qualidade de Dados e Governança em Produção.

## Contexto do problema

Uma fintech de concessão de crédito possui um modelo de Credit Scoring em
produção. Mudanças recentes na economia (inflação, novos perfis de clientes)
levantam a suspeita de que o modelo está sofrendo **degradação silenciosa**.
Este projeto constrói a camada de **Sustentação e Confiabilidade** desse
modelo: contratos de qualidade de dados, simulação e detecção de Data/Concept
Drift, observabilidade do pipeline e documentação de governança (LGPD).

## Sumário

- [Dataset](#dataset)
- [Como rodar o projeto](#como-rodar-o-projeto) <!-- preenchido na Etapa 3 -->
- [Etapa 1 — Validação de Dados e Contratos](#etapa-1--validação-de-dados-e-contratos) <!-- preenchido na Etapa 1 -->
- [Etapa 2 — Simulação e Detecção de Drift](#etapa-2--simulação-e-detecção-de-drift) <!-- preenchido na Etapa 2 -->
- [Etapa 3 — Observabilidade de Pipelines e Modelos](#etapa-3--observabilidade-de-pipelines-e-modelos) <!-- preenchido na Etapa 3 -->
- [Etapa 4 — Governança e LGPD](#etapa-4--governança-e-lgpd) <!-- preenchido na Etapa 4 -->
- [Vídeo STAR](#vídeo-star) <!-- preenchido na Etapa 4 -->

## Dataset

Base: **Home Credit Default Risk** (Kaggle) — `application_train.csv`,
classificação binária (`TARGET`: cliente com dificuldade de pagamento ou não),
~307 mil linhas, bem acima do mínimo de 5.000 amostras exigido. Dele são
extraídas e amostradas as colunas relevantes para o problema de risco de
crédito (idade, renda, tempo de emprego, valor do crédito, status familiar,
educação).

Passo a passo para reproduzir o dataset de referência localmente:

```bash
poetry install
poetry run python -m credit_monitoring.data.download     # baixa application_train.csv para data/raw/
poetry run python -m credit_monitoring.data.preprocess    # gera data/reference/credit_reference.csv
```

Ver [`download.py`](src/credit_monitoring/data/download.py) (instruções/chamada à Kaggle API) e
[`preprocess.py`](src/credit_monitoring/data/preprocess.py) (seleção de colunas, limpeza e amostragem estratificada por `target`).
O resultado (`data/reference/credit_reference.csv`) já é versionado no repositório, então rodar o download não é necessário só para explorar o projeto.

## Estrutura do projeto

```
src/credit_monitoring/
├── data/         # download, pré-processamento e contratos de qualidade (Pandera)
├── model/        # treino e inferência do classificador baseline
├── drift/        # simulação de drift e detecção (Evidently AI)
└── monitoring/   # logging, MLflow, métricas Prometheus e orquestração do pipeline
scripts/          # CLIs de entrada (pipeline completo, demo de bloqueio de dados ruins)
reports/          # relatórios HTML de drift e dashboard de observabilidade
tests/            # testes automatizados (pytest)
```

## Como rodar o projeto

Com o dataset de referência já versionado em `data/reference/` (ver
[Dataset](#dataset)), o pipeline completo roda ponta a ponta com:

```bash
poetry install
poetry run python scripts/run_pipeline.py
```

Isso executa, em sequência, **validação de dados → treino do modelo →
predição/scoring em produção → detecção de drift**, e produz:

- `data/model/baseline_model.joblib` — pipeline treinado (pré-processamento + XGBoost).
- `reports/baseline_model_metrics.json`, `reports/score_comparison.json` — métricas do modelo e comparação de scores.
- `reports/drift/drift_report.html` e `drift_summary.md` — relatório visual e resumo de drift (Evidently).
- `reports/dashboard/dashboard.html` — **dashboard consolidado** com o resultado de cada etapa, métricas do modelo, comparação de scores e drift.
- `logs/pipeline.log` e `logs/pipeline_events.jsonl` — log estruturado de cada execução.
- `mlruns/` — experimentos registrados no MLflow (`mlflow ui` para inspecionar).
- `metrics/pipeline_metrics.prom` — métricas no formato Prometheus (textfile collector).

## Etapa 3 — Observabilidade de Pipelines e Modelos

A observabilidade do pipeline combina três mecanismos complementares,
orquestrados por [`monitoring/pipeline_runner.py`](src/credit_monitoring/monitoring/pipeline_runner.py):

1. **Logging estruturado** ([`monitoring/logging_config.py`](src/credit_monitoring/monitoring/logging_config.py)) — cada etapa
   (`validacao_dados`, `treino_modelo`, `predicao`, `deteccao_drift`) é
   registrada como um evento em JSON Lines (`logs/pipeline_events.jsonl`),
   com status (`success`/`failure`), duração e metadados. Falhas são
   relançadas após o registro, então um erro em qualquer etapa interrompe o
   pipeline de forma visível.
2. **Rastreamento de experimentos (MLflow)** — cada etapa roda dentro de um
   MLflow run, registrando parâmetros do modelo, métricas (accuracy,
   ROC-AUC, F1, colunas com drift) e artefatos (modelo serializado,
   relatórios de drift) — permitindo comparar execuções ao longo do tempo.
3. **Métricas Prometheus** ([`monitoring/metrics.py`](src/credit_monitoring/monitoring/metrics.py)) — expostas no
   padrão *textfile collector* (`metrics/pipeline_metrics.prom`), já que o
   pipeline roda em lote, não como serviço de longa duração:
   - `pipeline_stage_runs_total{stage,status}` — nº de execuções de cada etapa, por status (saúde operacional do pipeline).
   - `pipeline_stage_duration_seconds{stage}` — latência de cada etapa (histograma).
   - `pipeline_rows_processed{stage}` — volume de linhas processadas na última execução de cada etapa.
   - `pipeline_last_run_timestamp_seconds{stage,status}` — quando cada etapa rodou por último (detecta pipelines parados/"silenciosos").
   - `pipeline_drift_alerts_total` — nº de execuções em que drift foi detectado no dataset de produção (alerta de degradação do modelo).

O **dashboard HTML consolidado** (`reports/dashboard/dashboard.html`, gerado por
[`monitoring/dashboard.py`](src/credit_monitoring/monitoring/dashboard.py)) reúne, numa única página, os eventos da
última execução, as métricas do modelo, a comparação de scores
referência/produção e o resumo de drift — hoje espalhados entre `logs/`,
`mlruns/`, `metrics/` e `reports/drift/`.

---

*As seções restantes (Governança/LGPD) serão preenchidas na Etapa 4.*
