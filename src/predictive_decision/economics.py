"""Profit, break-even and risk for selecting the top sites by predicted volume."""

import numpy as np
import pandas as pd


def break_even_volume(budget: float, revenue_per_unit: float, n_select: int) -> float:
    """Average volume per selected site needed to recover the budget."""
    return budget / (n_select * revenue_per_unit)


def profit_top_sites(validation: pd.DataFrame, n_select: int,
                     budget: float, revenue_per_unit: float) -> float:
    """Profit from the n sites with the highest predicted volume, using their actual volumes."""
    top = validation.nlargest(n_select, "predicted")
    return float(top["actual"].sum() * revenue_per_unit - budget)


def bootstrap_profit(validation: pd.DataFrame, n_select: int, budget: float,
                     revenue_per_unit: float, n_samples: int = 1000, seed: int = 0) -> dict:
    """Estimate profit distribution by sampling sites with replacement and selecting the top n each time."""
    rng = np.random.default_rng(seed)
    profits = np.empty(n_samples)
    for i in range(n_samples):
        sample = validation.sample(n=len(validation), replace=True,
                                   random_state=int(rng.integers(1 << 31)))
        profits[i] = profit_top_sites(sample, n_select, budget, revenue_per_unit)
    return {
        "mean_profit": float(profits.mean()),
        "ci_low": float(np.quantile(profits, 0.025)),
        "ci_high": float(np.quantile(profits, 0.975)),
        "loss_risk": float((profits < 0).mean()),
    }
