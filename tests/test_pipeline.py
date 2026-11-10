import numpy as np
import pandas as pd
import pytest

from predictive_decision.data import REGION_NAMES, generate_regions
from predictive_decision.economics import bootstrap_profit, break_even_volume, profit_top_sites
from predictive_decision.model import FEATURES, train_region


def test_generate_regions_shape_and_names():
    regions = generate_regions(n_sites=500, seed=1)
    assert tuple(regions) == REGION_NAMES
    for df in regions.values():
        assert list(df.columns) == FEATURES + ["product"]
        assert len(df) == 500
        assert (df["product"] >= 0).all()


def test_train_region_split_and_metrics():
    df = generate_regions(n_sites=2000, seed=2)["region_a"]
    result = train_region(df, seed=2)
    assert len(result["validation"]) == 500
    assert result["rmse"] > 0
    assert result["mean_predicted"] == pytest.approx(result["validation"]["predicted"].mean())


def test_break_even_volume():
    assert break_even_volume(100.0, 10.0, 5) == pytest.approx(2.0)


def test_profit_top_sites_selects_highest_predicted():
    validation = pd.DataFrame({
        "actual": [10.0, 1.0, 20.0],
        "predicted": [9.0, 0.5, 19.0],
    })
    profit = profit_top_sites(validation, n_select=2, budget=100.0, revenue_per_unit=10.0)
    assert profit == pytest.approx((10.0 + 20.0) * 10.0 - 100.0)


def test_bootstrap_profit_is_reproducible():
    validation = generate_regions(n_sites=400, seed=3)["region_b"].assign(
        actual=lambda d: d["product"], predicted=lambda d: d["product"])
    a = bootstrap_profit(validation, 50, 1_000.0, 100.0, n_samples=50, seed=7)
    b = bootstrap_profit(validation, 50, 1_000.0, 100.0, n_samples=50, seed=7)
    assert a == b
    assert 0.0 <= a["loss_risk"] <= 1.0
    assert np.isfinite(a["mean_profit"])
