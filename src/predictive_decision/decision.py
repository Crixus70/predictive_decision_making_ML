"""Choose price and weekly marketing per product that maximize profit over a horizon."""

import itertools

import numpy as np
import pandas as pd

from predictive_decision.data import BASE_PRICE, PRODUCTS, UNIT_COST, base_demand, expected_units
from predictive_decision.forecast import predict_units

PRICE_GRID = np.arange(48.0, 80.1, 2.0)
MARKETING_GRID = np.arange(0.0, 501.0, 50.0)
CURRENT_POLICY = {"price": BASE_PRICE, "marketing": 100.0}


def optimize(model, weeks) -> dict:
    """Best (price, marketing) per product using the forecast model over the given weeks."""
    policy = {}
    for product in PRODUCTS:
        best_profit, best_price, best_mkt = -np.inf, None, None
        for price, mkt in itertools.product(PRICE_GRID, MARKETING_GRID):
            grid = pd.DataFrame({"product": product, "week": weeks, "price": price, "marketing": mkt})
            total = (predict_units(model, grid) * (price - UNIT_COST) - mkt).sum()
            if total > best_profit:
                best_profit, best_price, best_mkt = total, price, mkt
        policy[product] = {"price": float(best_price), "marketing": float(best_mkt)}
    return policy


def marketing_roi(policy: dict, weeks) -> pd.Series:
    """Incremental gross margin from marketing divided by marketing spend, per product."""
    roi = {}
    for i, product in enumerate(PRODUCTS):
        d = policy[product]
        base = base_demand(i)
        with_mkt = expected_units(base, d["price"], d["marketing"], weeks)
        without = expected_units(base, d["price"], 0.0, weeks)
        spend = d["marketing"] * len(weeks)
        incremental = ((with_mkt - without) * (d["price"] - UNIT_COST)).sum()
        roi[product] = incremental / spend if spend > 0 else np.nan
    return pd.Series(roi, name="marketing_roi")


def expected_profit(policy: dict, weeks) -> float:
    """Profit under the true expected demand, summed over the horizon."""
    total = 0.0
    for i, product in enumerate(PRODUCTS):
        d = policy[product]
        units = expected_units(base_demand(i), d["price"], d["marketing"], weeks)
        total += (units * (d["price"] - UNIT_COST) - d["marketing"]).sum()
    return float(total)
