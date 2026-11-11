import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from credit_monitoring.data.contracts import validate_dataset, validation_report


def test_valid_dataset_passes_contract(synthetic_reference_df):
    validated = validate_dataset(synthetic_reference_df)
    assert len(validated) == len(synthetic_reference_df)


def test_bad_batch_is_rejected_by_contract(synthetic_reference_df):
    bad_df = synthetic_reference_df.copy()

    # Regra 1: idade minima (menor de idade no lote).
    bad_df.loc[0, "age"] = 15.0
    # Regra 2: renda nao pode ser nula.
    bad_df.loc[1, "income"] = None
    # Regra 3: sem duplicatas de client_id.
    bad_df.loc[2, "client_id"] = bad_df.loc[3, "client_id"]
    # Extra: valor monetario invalido (negativo).
    bad_df.loc[4, "credit_amount"] = -100.0

    with pytest.raises(SchemaErrors):
        validate_dataset(bad_df)

    failure_cases = validation_report(bad_df)
    assert failure_cases is not None

    failed_columns = set(failure_cases["column"].dropna())
    assert {"age", "income", "client_id", "credit_amount"}.issubset(failed_columns)


def test_validation_report_returns_none_for_valid_dataset(synthetic_reference_df):
    assert validation_report(synthetic_reference_df) is None
