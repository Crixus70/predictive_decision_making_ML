"""Simulated A/B test comparing two pricing policies across stores."""

import math

import numpy as np

from predictive_decision.data import PRODUCTS, UNIT_COST, base_demand, expected_units


def store_profits(policy: dict, weeks, n_stores: int, rng: np.random.Generator) -> np.ndarray:
    """Total profit per store over the horizon, with store-level demand differences."""
    scale = rng.lognormal(0.0, 0.3, n_stores)
    totals = np.zeros(n_stores)
    for i, product in enumerate(PRODUCTS):
        d = policy[product]
        mean = expected_units(base_demand(i), d["price"], d["marketing"], weeks)
        units = rng.poisson(np.outer(scale, mean))
        totals += (units * (d["price"] - UNIT_COST)).sum(axis=1) - d["marketing"] * len(weeks)
    return totals


def ab_test(control: np.ndarray, treatment: np.ndarray) -> dict:
    """Relative uplift of treatment over control with a 95% CI and a two-sided p-value."""
    diff = treatment.mean() - control.mean()
    se = math.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))
    z = diff / se
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    base = abs(control.mean())
    return {
        "uplift_pct": 100 * diff / base,
        "ci_low_pct": 100 * (diff - 1.96 * se) / base,
        "ci_high_pct": 100 * (diff + 1.96 * se) / base,
        "p_value": p_value,
    }


def sample_size_per_arm(control: np.ndarray, mde_pct: float = 5.0) -> int:
    """Stores per arm to detect a relative uplift of mde_pct (95% confidence, 80% power)."""
    cv = control.std(ddof=1) / abs(control.mean())
    delta = mde_pct / 100
    return math.ceil(2 * (1.96 + 0.84) ** 2 * cv ** 2 / delta ** 2)
