"""
Unit & Integration Tests for Stratified Feature Sampling & Inverse Probability Weighting.
"""

import numpy as np
import pandas as pd
import pytest

from src.fusion.sampling import build_stratified_sampling_sql, load_feature_mart_sample


@pytest.fixture
def dummy_mart_parquet(tmp_path):
    """Creates a temporary feature mart with known imbalanced strata."""
    records = []
    # 900 dry, 80 light rain, 20 heavy rain
    for i in range(1000):
        if i < 900:
            p = 0.0
        elif i < 980:
            p = 0.5
        else:
            p = 2.0

        records.append(
            {
                "trip_id": f"trip_{i}",
                "stop_sequence": i % 10,
                "route_id": "100" if i % 2 == 0 else "1",
                "transit_cohort": "urban_bus" if i % 2 == 0 else "urban_tram",
                "is_tram": 0 if i % 2 == 0 else 1,
                "is_dedicated_right_of_way": False if i % 2 == 0 else True,
                "is_peak_hour": 1 if i % 3 == 0 else 0,
                "arrival_ts": f"2026-10-20 0{i % 8 + 1}:00:00",
                "arrival_delay_seconds": float(i % 120),
                "delta_t_run": float(i % 60),
                "precipitation_mm": p,
            }
        )
    df = pd.DataFrame(records)
    out_file = tmp_path / "test_mart.parquet"
    df.to_parquet(out_file)
    return str(out_file)


def test_build_stratified_sampling_sql_structure(dummy_mart_parquet):
    sql = build_stratified_sampling_sql(dummy_mart_parquet, sample_size=200)
    assert "WITH base_data AS" in sql
    assert "inverse_sampling_weight" in sql
    assert "mod_heavy_rain" in sql


def test_load_stratified_sample_weights_normalized(dummy_mart_parquet):
    df_sample = load_feature_mart_sample(dummy_mart_parquet, sample_size=200, strategy="stratified")
    assert not df_sample.empty
    assert len(df_sample) <= 250
    assert "inverse_sampling_weight" in df_sample.columns

    # Check that weights are positive and normalized around 1.0
    weights = df_sample["inverse_sampling_weight"].to_numpy()
    assert np.all(weights > 0)
    assert np.isclose(weights.mean(), 1.0, atol=0.05)


def test_load_stratified_sample_balances_rare_regimes(dummy_mart_parquet):
    df_sample = load_feature_mart_sample(dummy_mart_parquet, sample_size=200, strategy="stratified")
    # In population, heavy rain is only 2%. In stratified sample, it should have substantial quota.
    heavy_rain_sampled = (df_sample["precipitation_mm"] > 1.0).sum()
    assert heavy_rain_sampled > 5  # Substantially boosted compared to 2% (which would be 4 in 200)


def test_load_sample_full_dataset(dummy_mart_parquet):
    df_full = load_feature_mart_sample(dummy_mart_parquet, sample_size=0, strategy="none")
    assert len(df_full) == 1000
    assert np.all(df_full["inverse_sampling_weight"] == 1.0)


def test_load_sample_uniform(dummy_mart_parquet):
    df_uni = load_feature_mart_sample(dummy_mart_parquet, sample_size=100, strategy="uniform")
    assert len(df_uni) == 100
    assert "inverse_sampling_weight" in df_uni.columns
    assert np.all(df_uni["inverse_sampling_weight"] == 1.0)
