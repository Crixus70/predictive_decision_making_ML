"""Run the full study: forecast validation, price and marketing decision, and simulated A/B test."""

import numpy as np
import pandas as pd

from predictive_decision.data import N_WEEKS, PRODUCTS, generate_sales
from predictive_decision.decision import CURRENT_POLICY, expected_profit, marketing_roi, optimize
from predictive_decision.experiment import ab_test, sample_size_per_arm, store_profits
from predictive_decision.forecast import fit_demand_model, validate

VALIDATION_START = 104
DECISION_START = 130
N_STORES = 200


def run(seed: int = 0) -> dict:
    sales = generate_sales(seed)
    forecast = validate(sales, VALIDATION_START, DECISION_START)

    model = fit_demand_model(sales[sales["week"] < DECISION_START])
    horizon = np.arange(DECISION_START, N_WEEKS)
    policy = optimize(model, horizon)
    current = {p: dict(CURRENT_POLICY) for p in PRODUCTS}

    rng = np.random.default_rng(seed)
    control = store_profits(current, horizon, N_STORES, rng)
    treatment = store_profits(policy, horizon, N_STORES, rng)

    return {
        "sales": sales,
        "forecast": forecast,
        "expected_profit_current": expected_profit(current, horizon),
        "expected_profit_new": expected_profit(policy, horizon),
        "ab": ab_test(control, treatment),
        "sample_size_per_arm": sample_size_per_arm(control),
        "control": control,
        "treatment": treatment,
        "policy": pd.DataFrame(policy).T,
        "roi": marketing_roi(policy, horizon),
        "horizon": horizon,
    }


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    result = run()
    f = result["forecast"]
    ab = result["ab"]
    print(f"Forecast MAE (units/week): model {f['mae_model']:.2f} vs naive {f['mae_naive']:.2f}")
    print(f"Expected profit, current policy: {result['expected_profit_current']:,.0f}")
    print(f"Expected profit, new policy:     {result['expected_profit_new']:,.0f}")
    print(f"A/B uplift: {ab['uplift_pct']:.2f}% "
          f"(95% CI {ab['ci_low_pct']:.2f}% to {ab['ci_high_pct']:.2f}%, p={ab['p_value']:.4f})")
    print(f"Stores per arm to detect a 5% uplift: {result['sample_size_per_arm']}")
    print()
    print(result["policy"].join(result["roi"]).to_string())
