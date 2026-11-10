import numpy as np

from predictive_decision.data import N_WEEKS, PRODUCTS, generate_sales
from predictive_decision.decision import CURRENT_POLICY, PRICE_GRID, expected_profit, marketing_roi, optimize
from predictive_decision.experiment import ab_test, store_profits
from predictive_decision.forecast import design, fit_demand_model, validate

HORIZON = np.arange(130, N_WEEKS)


def test_sales_shape_and_signs():
    sales = generate_sales(0)
    assert len(sales) == len(PRODUCTS) * N_WEEKS
    assert (sales["units"] >= 0).all()
    assert sales["price"].between(0.8 * 60, 1.2 * 60).all()


def test_sales_are_reproducible():
    assert generate_sales(0).equals(generate_sales(0))


def test_model_beats_naive_forecast():
    result = validate(generate_sales(0), 104, 130)
    assert result["mae_model"] < result["mae_naive"]


def test_design_matrix_has_product_indicators():
    x = design(generate_sales(0).head(5))
    assert x.shape == (5, len(PRODUCTS) + 3)


def test_optimized_policy_beats_current_policy():
    sales = generate_sales(0)
    model = fit_demand_model(sales[sales["week"] < 130])
    policy = optimize(model, HORIZON)
    current = {p: dict(CURRENT_POLICY) for p in PRODUCTS}
    assert expected_profit(policy, HORIZON) > expected_profit(current, HORIZON)
    assert all(PRICE_GRID.min() <= v["price"] <= PRICE_GRID.max() for v in policy.values())


def test_marketing_roi_is_positive_where_spend_exists():
    policy = {p: {"price": 66.0, "marketing": 200.0} for p in PRODUCTS}
    roi = marketing_roi(policy, HORIZON)
    assert roi.notna().all()
    assert (roi > 0).all()


def test_ab_test_detects_clear_uplift():
    rng = np.random.default_rng(0)
    control = rng.normal(100, 10, 300)
    treatment = rng.normal(115, 10, 300)
    result = ab_test(control, treatment)
    assert result["p_value"] < 0.001
    assert result["ci_low_pct"] > 0


def test_store_profits_are_reproducible():
    policy = {p: dict(CURRENT_POLICY) for p in PRODUCTS}
    a = store_profits(policy, HORIZON, 50, np.random.default_rng(1))
    b = store_profits(policy, HORIZON, 50, np.random.default_rng(1))
    assert np.array_equal(a, b)
