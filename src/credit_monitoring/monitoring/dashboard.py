"""Monta o dashboard HTML consolidado do pipeline (Etapa 3.4).

Reune, numa unica pagina estatica, o resultado da ultima execucao do
pipeline: eventos de cada etapa (sucesso/falha/duracao), metricas do modelo
treinado, comparacao de scores entre referencia e producao, e o resumo de
drift — hoje espalhados entre `logs/`, `mlruns/`, `metrics/` e
`reports/drift/`.

Uso:
    poetry run python scripts/run_pipeline.py
"""

from __future__ import annotations

from datetime import datetime, timezone

from credit_monitoring.utils.paths import DASHBOARD_DIR

DASHBOARD_HTML_FILE = DASHBOARD_DIR / "dashboard.html"


def _format_events_table(events: list[dict]) -> str:
    if not events:
        return "<p>Nenhum evento registrado.</p>"

    rows = []
    for event in events:
        rows.append(
            "<tr>"
            f"<td>{event['stage']}</td>"
            f"<td class=\"{event['status']}\">{event['status']}</td>"
            f"<td>{event['duration_seconds']:.2f}s</td>"
            f"<td>{event['timestamp']}</td>"
            "</tr>"
        )
    body = "".join(rows)
    return (
        "<table><thead><tr><th>Etapa</th><th>Status</th><th>Duracao</th>"
        f"<th>Timestamp</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _format_drift_table(drifted_columns: list[dict]) -> str:
    if not drifted_columns:
        return "<p>Nenhuma coluna com drift detectado.</p>"

    rows = []
    for column in drifted_columns:
        rows.append(
            "<tr>"
            f"<td>{column['column']}</td>"
            f"<td>{column['column_type']}</td>"
            f"<td>{column['stattest_name']}</td>"
            f"<td>{column['drift_score']:.4f}</td>"
            "</tr>"
        )
    body = "".join(rows)
    return (
        "<table><thead><tr><th>Coluna</th><th>Tipo</th><th>Teste</th>"
        f"<th>Score</th></tr></thead><tbody>{body}</tbody></table>"
    )


def build_dashboard_html(pipeline_result: dict) -> str:
    """Monta a pagina HTML a partir do dict retornado por `run_full_pipeline()`."""
    training_metrics = pipeline_result["training_metrics"]
    comparison = pipeline_result["prediction_comparison"]
    drift_summary = pipeline_result["drift_summary"]
    events = pipeline_result["events"]

    generated_at = datetime.now(timezone.utc).isoformat()

    if drift_summary["dataset_drift_detected"]:
        drift_alert = (
            "<div class=\"alert\">Drift detectado: "
            f"{drift_summary['number_of_drifted_columns']}/{drift_summary['number_of_columns']} "
            f"colunas ({drift_summary['share_of_drifted_columns'] * 100:.1f}%).</div>"
        )
    else:
        drift_alert = "<p>Nenhum drift significativo detectado no dataset.</p>"

    events_table = _format_events_table(events)
    drift_table = _format_drift_table(drift_summary["drifted_columns"])

    return f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Dashboard de Observabilidade - Credit Scoring</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
h1, h2 {{ color: #0b3d91; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
th {{ background: #f0f0f0; }}
.success {{ color: #1a7f37; font-weight: bold; }}
.failure {{ color: #c0392b; font-weight: bold; }}
.alert {{ background: #fff3cd; border: 1px solid #f0ad4e; padding: 0.8rem; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Dashboard de Observabilidade &mdash; Credit Scoring</h1>
<p>Gerado em {generated_at}</p>

<h2>Execucao do pipeline</h2>
{events_table}

<h2>Modelo baseline</h2>
<table>
<tr><th>Accuracy</th><td>{training_metrics['accuracy']:.4f}</td></tr>
<tr><th>ROC-AUC</th><td>{training_metrics['roc_auc']:.4f}</td></tr>
<tr><th>F1</th><td>{training_metrics['f1']:.4f}</td></tr>
<tr><th>Amostras de teste</th><td>{training_metrics['n_test_samples']}</td></tr>
</table>

<h2>Scores: referencia vs producao</h2>
<table>
<tr><th></th><th>Referencia</th><th>Producao</th><th>Delta</th></tr>
<tr><th>Score medio</th><td>{comparison['reference']['mean_score']:.4f}</td><td>{comparison['production']['mean_score']:.4f}</td><td>{comparison['mean_score_delta']:+.4f}</td></tr>
<tr><th>% alto risco</th><td>{comparison['reference']['pct_high_risk']:.1f}%</td><td>{comparison['production']['pct_high_risk']:.1f}%</td><td>{comparison['pct_high_risk_delta']:+.1f} pp</td></tr>
</table>

<h2>Deteccao de drift</h2>
{drift_alert}
{drift_table}
<p>Relatorio completo do Evidently: <a href="../drift/drift_report.html">drift_report.html</a></p>

</body>
</html>
"""


def save_dashboard(pipeline_result: dict, out_path=DASHBOARD_HTML_FILE) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_dashboard_html(pipeline_result), encoding="utf-8")
