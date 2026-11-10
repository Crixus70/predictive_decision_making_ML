"""Synthetic weekly retail sales for 20 products with price and marketing decisions."""

import numpy as np
import pandas as pd

PRODUCTS = [f"p{i:02d}" for i in range(1, 21)]
N_WEEKS = 156
UNIT_COST = 40.0
BASE_PRICE = 60.0
ELASTICITY = -2.5
MARKETING_LIFT = 0.4
MARKETING_HALF_SATURATION = 150.0


def base_demand(product_index: int) -> float:
    """Deterministic baseline demand per product, spread between 40 and 200 units."""
    return 40.0 + 160.0 * ((product_index * 37) % 20) / 19


def expected_units(base, price, marketing, week):
    """True expected weekly units: price elasticity, saturating marketing lift and seasonality."""
    season = 0.2 * np.sin(2 * np.pi * np.asarray(week) / 52)
    price_effect = (np.asarray(price) / BASE_PRICE) ** ELASTICITY
    m = np.asarray(marketing, dtype=float)
    marketing_effect = 1 + MARKETING_LIFT * m / (m + MARKETING_HALF_SATURATION)
    return base * np.exp(season) * price_effect * marketing_effect


def generate_sales(seed: int = 0) -> pd.DataFrame:
    """Return one row per product and week with price, marketing and realized units."""
    rng = np.random.default_rng(seed)
    weeks = np.arange(N_WEEKS)
    frames = []
    for i, product in enumerate(PRODUCTS):
        price = BASE_PRICE * rng.uniform(0.8, 1.2, N_WEEKS)
        marketing = rng.uniform(0.0, 500.0, N_WEEKS)
        mean = expected_units(base_demand(i), price, marketing, weeks)
        frames.append(pd.DataFrame({
            "product": product,
            "week": weeks,
            "price": price.round(2),
            "marketing": marketing.round(2),
            "units": rng.poisson(mean),
        }))
    return pd.concat(frames, ignore_index=True)
