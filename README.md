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

## Etapa 4 — Governança e LGPD

### Dados pessoais tratados e minimização

O dataset de referência ([`preprocess.py`](src/credit_monitoring/data/preprocess.py))
já aplica minimização de dados na origem: do CSV bruto do Home Credit (122
colunas) são extraídas apenas **17 colunas** diretamente relevantes para o
risco de crédito (renda, valor do crédito, idade, tempo de emprego,
escolaridade, estado civil, tipo de moradia, ocupação, etc.). Não há nome,
CPF, endereço, telefone ou e-mail em nenhuma etapa do pipeline — `client_id`
é apenas a chave substituta (`SK_ID_CURR`) já anonimizada pela fonte (Kaggle),
sem relação com documentos reais.

Dentre as colunas mantidas, `gender` é **dado pessoal** mas não se qualifica
como **dado sensível** pela enumeração taxativa do Art. 5º, II da LGPD
(origem racial/étnica, convicção religiosa, opinião política, dado referente
à saúde/vida sexual, dado genético ou biométrico). Ainda assim, por ser um
atributo com potencial discriminatório, seu uso é tratado com atenção
especial na seção de [mitigação de viés](#mitigação-de-viés-e-discriminação-algorítmica)
abaixo, em vez de ser simplesmente removido — removê-lo sem monitorar não
eliminaria um eventual viés codificado em variáveis correlacionadas
(`education`, `occupation_type`, `organization_type`).

### Base legal para a decisão de crédito

O tratamento dos dados para o scoring de crédito se apoia no **Art. 7º, X da
LGPD** — tratamento de dados pessoais para "proteção do crédito", hipótese
prevista especificamente para esse caso de uso, sem depender do consentimento
individual do titular a cada concessão. Isso não dispensa os demais deveres
da LGPD: transparência sobre os critérios gerais da decisão (Art. 20, §1º) e
possibilidade de solicitação de revisão (próxima seção).

### Direito à revisão humana das decisões automatizadas

O Art. 20 da LGPD garante ao titular o direito de solicitar **revisão de
decisões tomadas unicamente com base em tratamento automatizado** de dados
pessoais que afetem seus interesses — o exemplo canônico da lei é justamente
a concessão de crédito. Na prática operacional isso significa que o score
produzido por este pipeline (`predicted_proba` / `predicted_label` em
[`predict.py`](src/credit_monitoring/model/predict.py)) deve alimentar uma
decisão sujeita a contestação e revisão humana, nunca ser a palavra final e
automática sobre a concessão do crédito.

### Retenção e descarte

- **Dado bruto** (`data/raw/application_train.csv`, baixado do Kaggle) nunca é
  versionado (`.gitignore`) e deve ser descartado localmente assim que o
  dataset de referência for gerado — não há necessidade de retê-lo.
- **Dataset de referência** (`data/reference/credit_reference.csv`) é uma
  amostra agregada (~25 mil linhas, sem identificadores reais) mantida apenas
  enquanto for necessária para treinar/auditar o modelo em produção; numa
  implantação real, deveria seguir o mesmo prazo de guarda de dados usados em
  decisão de crédito exigido pelo regulador do setor (ex.: Bacen), e ser
  reamostrada/descartada ao final desse ciclo.
- **Dataset de produção simulado** (`data/production/`) e as predições
  derivadas são artefatos de monitoramento, não dados reais de clientes —
  não há titular a proteger nesse caso, mas o mesmo prazo de retenção do
  dataset de referência é aplicado por consistência.
- **Artefatos de observabilidade** (`logs/`, `mlruns/`, `metrics/`) guardam
  apenas metadados operacionais do pipeline (status, duração, métricas
  agregadas) — nenhum dado pessoal individualizado circula por esses
  mecanismos.

### Mitigação de viés e discriminação algorítmica

O dataset de referência carrega atributos que podem servir de proxy para
discriminação indireta (`gender`, `education`, `family_status`,
`housing_type`, `occupation_type`, `organization_type`, `region_rating`).
Duas frentes deste projeto já apoiam a mitigação de viés, e podem ser
estendidas:

1. **Monitoramento de drift por feature** ([`drift/detect.py`](src/credit_monitoring/drift/detect.py))
   já quebra o resultado por coluna — o mesmo mecanismo usado para detectar
   degradação do modelo serve para detectar quando a distribuição de um
   atributo sensível/proxy muda de forma desproporcional em relação às
   demais, um sinal de alerta precoce de viés introduzido por mudança na
   população de entrada.
2. **Métricas de score desagregadas**: o `score_comparison.json` gerado por
   [`predict.py`](src/credit_monitoring/model/predict.py) compara a
   distribuição de scores entre referência e produção; o próximo passo
   natural (fora do escopo deste desafio) é desagregar essa mesma comparação
   por grupo (ex.: por `gender` ou faixa de `education`) para detectar
   impacto desproporcional (*disparate impact*) antes de um novo modelo ir
   para produção.

---

*A seção [Vídeo STAR](#vídeo-star) será preenchida na consolidação final do README.*
