"""Download do dataset bruto "Home Credit Default Risk" (Kaggle).

O arquivo `application_train.csv` (~170MB, ~307k linhas) nao e versionado no
repositorio (ver .gitignore) — cada pessoa baixa a propria copia localmente.

Pre-requisito: ter a Kaggle CLI configurada com credenciais proprias
(arquivo `~/.kaggle/kaggle.json`, gerado em https://www.kaggle.com -> Account
-> Create New API Token). Este script NAO lida com credenciais, apenas chama
a API ja configurada pelo usuario.

Uso:
    poetry run python -m credit_monitoring.data.download

Caso prefira baixar manualmente, acesse a competicao
https://www.kaggle.com/c/home-credit-default-risk, aceite as regras, baixe
`application_train.csv` e salve em `data/raw/application_train.csv`.
"""

from __future__ import annotations

import logging
import zipfile

from credit_monitoring.utils.paths import RAW_APPLICATION_FILE, RAW_DIR

logger = logging.getLogger(__name__)

COMPETITION = "home-credit-default-risk"
EXPECTED_MEMBER = "application_train.csv"


def download_home_credit_dataset(force: bool = False) -> None:
    """Baixa application_train.csv da competicao Home Credit para data/raw/.

    Requer a biblioteca `kaggle` instalada e credenciais configuradas pelo
    usuario (nao providas por este script).
    """
    if RAW_APPLICATION_FILE.exists() and not force:
        logger.info("Arquivo ja existe em %s — nada a fazer.", RAW_APPLICATION_FILE)
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - dependencia opcional
        raise RuntimeError(
            "Pacote 'kaggle' nao instalado. Rode 'poetry install' ou baixe o "
            "CSV manualmente em https://www.kaggle.com/c/home-credit-default-risk "
            "e salve em data/raw/application_train.csv."
        ) from exc

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()  # usa ~/.kaggle/kaggle.json do proprio usuario

    logger.info("Baixando dataset da competicao '%s'...", COMPETITION)
    api.competition_download_file(COMPETITION, EXPECTED_MEMBER, path=str(RAW_DIR))

    zip_path = RAW_DIR / f"{EXPECTED_MEMBER}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    if not RAW_APPLICATION_FILE.exists():
        raise RuntimeError(
            f"Download concluido mas {RAW_APPLICATION_FILE} nao foi encontrado. "
            "Verifique se voce aceitou as regras da competicao no site do Kaggle."
        )

    logger.info("Dataset salvo em %s", RAW_APPLICATION_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_home_credit_dataset()
