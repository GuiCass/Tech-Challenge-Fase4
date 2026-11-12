"""Resumo customizado (Markdown) do drift detectado, legivel por humanos.

Complementa o HTML denso/tecnico do Evidently (`detect.py`) com uma tabela
enxuta apontando CLARAMENTE quais features degradaram e com qual severidade
— o entregavel pedido na Etapa 2 do desafio.
"""

from __future__ import annotations

from credit_monitoring.utils.paths import DRIFT_REPORTS_DIR

DRIFT_SUMMARY_MARKDOWN_FILE = DRIFT_REPORTS_DIR / "drift_summary.md"

# Limiares de severidade. PSI segue a convencao de mercado (>=0.25 severo,
# >=0.10 moderado). K-S usa p-valor (quanto menor, mais significativo o
# drift), entao os limiares sao invertidos em relacao ao PSI.
PSI_SEVERE_THRESHOLD = 0.25
PSI_MODERATE_THRESHOLD = 0.10

KS_PVALUE_SEVERE_THRESHOLD = 0.001
KS_PVALUE_MODERATE_THRESHOLD = 0.01


def classify_severity(column_type: str, drift_score: float) -> str:
    """Classifica a severidade do drift de uma coluna em leve/moderado/severo."""
    if column_type == "cat":
        if drift_score >= PSI_SEVERE_THRESHOLD:
            return "severo"
        if drift_score >= PSI_MODERATE_THRESHOLD:
            return "moderado"
        return "leve"

    # numerica (K-S p-valor): menor p-valor = drift mais significativo.
    if drift_score < KS_PVALUE_SEVERE_THRESHOLD:
        return "severo"
    if drift_score < KS_PVALUE_MODERATE_THRESHOLD:
        return "moderado"
    return "leve"


def build_drift_summary_markdown(summary: dict) -> str:
    """Monta o relatorio Markdown a partir do dict retornado por `detect.run_drift_detection`."""
    status = "SIM" if summary["dataset_drift_detected"] else "nao"
    lines = [
        "# Relatorio de Drift — Resumo Executivo",
        "",
        f"- Drift detectado no dataset como um todo: **{status}**",
        f"- Colunas com drift: **{summary['number_of_drifted_columns']}/"
        f"{summary['number_of_columns']}** "
        f"({summary['share_of_drifted_columns']:.1%})",
        "",
    ]

    if summary["drifted_columns"]:
        lines.append("| Feature | Tipo | Teste estatistico | Score | Severidade |")
        lines.append("|---|---|---|---|---|")
        for col in summary["drifted_columns"]:
            severity = classify_severity(col["column_type"], col["drift_score"])
            lines.append(
                f"| `{col['column']}` | {col['column_type']} | {col['stattest_name']} | "
                f"{col['drift_score']:.4g} | **{severity}** |"
            )
    else:
        lines.append("Nenhuma feature apresentou drift estatisticamente significativo.")

    lines.append("")
    return "\n".join(lines)


def save_drift_summary_markdown(summary: dict, out_path=DRIFT_SUMMARY_MARKDOWN_FILE) -> str:
    markdown = build_drift_summary_markdown(summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    return markdown
