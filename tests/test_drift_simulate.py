from credit_monitoring.data.contracts import validate_dataset
from credit_monitoring.drift.simulate import simulate_production_dataset


def test_simulate_production_dataset_keeps_schema_and_size(synthetic_reference_df):
    production_df = simulate_production_dataset(synthetic_reference_df)

    assert len(production_df) == len(synthetic_reference_df)
    assert set(production_df.columns) == set(synthetic_reference_df.columns)
    assert production_df["client_id"].is_unique


def test_simulate_production_dataset_still_passes_data_contract(synthetic_reference_df):
    production_df = simulate_production_dataset(synthetic_reference_df)

    # o drift muda a DISTRIBUICAO, nao a validade estrutural dos dados.
    validated = validate_dataset(production_df)
    assert len(validated) == len(production_df)


def test_simulate_production_dataset_injects_income_and_employment_drift(synthetic_reference_df):
    production_df = simulate_production_dataset(synthetic_reference_df, random_state=123)

    reference_income_mean = synthetic_reference_df["income"].mean()
    production_income_mean = production_df["income"].mean()
    assert production_income_mean < reference_income_mean

    reference_employment_mean = synthetic_reference_df["employment_years"].mean()
    production_employment_mean = production_df["employment_years"].mean()
    assert production_employment_mean < reference_employment_mean


def test_simulate_production_dataset_is_reproducible(synthetic_reference_df):
    first_run = simulate_production_dataset(synthetic_reference_df, random_state=7)
    second_run = simulate_production_dataset(synthetic_reference_df, random_state=7)

    assert first_run.equals(second_run)
