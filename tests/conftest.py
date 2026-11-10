import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_reference_df() -> pd.DataFrame:
    """Dataset sintetico com o mesmo schema do dataset de referencia real.

    Compartilhado entre os modulos de teste para nao depender do download
    real do Home Credit (Kaggle) ao rodar a suite de testes.
    """
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "client_id": range(n),
            "target": rng.integers(0, 2, size=n),
            "gender": rng.choice(["M", "F"], size=n),
            "num_children": rng.integers(0, 4, size=n),
            "income": rng.uniform(1_000, 20_000, size=n),
            "credit_amount": rng.uniform(5_000, 100_000, size=n),
            "annuity_amount": rng.uniform(500, 5_000, size=n),
            "education": rng.choice(["Secondary", "Higher education"], size=n),
            "family_status": rng.choice(["Married", "Single"], size=n),
            "housing_type": rng.choice(["House / apartment", "Rented apartment"], size=n),
            "age": rng.uniform(19, 70, size=n),
            "employment_years": rng.uniform(0, 30, size=n),
            "owns_car": rng.choice(["Y", "N"], size=n),
            "owns_realty": rng.choice(["Y", "N"], size=n),
            # inclui NaN de proposito para validar a imputacao
            "occupation_type": rng.choice(["Laborers", "Managers", None], size=n),
            "region_rating": rng.integers(1, 4, size=n),
            "organization_type": rng.choice(["Business Entity", "Self-employed"], size=n),
        }
    )
