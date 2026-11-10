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

Instruções de download e pré-processamento: ver [`src/credit_monitoring/data/download.py`](src/credit_monitoring/data/download.py).

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

---

*As seções abaixo serão preenchidas progressivamente ao longo das Etapas 1 a 4.*
