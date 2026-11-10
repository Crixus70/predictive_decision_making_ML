"""Run the full selection study on all regions and print a summary."""

import pandas as pd

from predictive_decision.data import generate_regions
from predictive_decision.economics import bootstrap_profit, break_even_volume, profit_top_sites
from predictive_decision.model import train_region

BUDGET = 100_000_000.0
REVENUE_PER_UNIT = 4_500.0
N_SELECT = 200


def run(seed: int = 0) -> pd.DataFrame:
    """Return one summary row per region."""
    rows = []
    for name, df in generate_regions(seed=seed).items():
        result = train_region(df, seed=seed)
        valid = result["validation"]
        rows.append({
            "region": name,
            "rmse": round(result["rmse"], 3),
            "mean_predicted": round(result["mean_predicted"], 2),
            "break_even_volume": round(break_even_volume(BUDGET, REVENUE_PER_UNIT, N_SELECT), 2),
            "profit": round(profit_top_sites(valid, N_SELECT, BUDGET, REVENUE_PER_UNIT), 2),
            **bootstrap_profit(valid, N_SELECT, BUDGET, REVENUE_PER_UNIT, seed=seed),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(run().to_string(index=False))
