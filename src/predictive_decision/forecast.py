"""Weekly demand forecast per product (Poisson regression) compared with a naive baseline."""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from predictive_decision.data import MARKETING_HALF_SATURATION, PRODUCTS


def design(df: pd.DataFrame) -> np.ndarray:
    """Product indicators, log price, saturated marketing spend and seasonality."""
    product_cols = [(df["product"] == p).to_numpy(float) for p in PRODUCTS]
    marketing = df["marketing"].to_numpy(float)
    extra = [
        np.log(df["price"].to_numpy(float)),
        marketing / (marketing + MARKETING_HALF_SATURATION),
        np.sin(2 * np.pi * df["week"].to_numpy(float) / 52),
    ]
    return np.column_stack(product_cols + extra)


def fit_demand_model(train: pd.DataFrame) -> PoissonRegressor:
    model = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=1000)
    model.fit(design(train), train["units"])
    return model


def predict_units(model: PoissonRegressor, df: pd.DataFrame) -> np.ndarray:
    return model.predict(design(df))


def naive_forecast(train: pd.DataFrame, test: pd.DataFrame, window: int = 4) -> np.ndarray:
    """Average units of the last `window` training weeks, per product."""
    recent = train[train["week"] >= train["week"].max() - window + 1]
    level = recent.groupby("product")["units"].mean()
    return test["product"].map(level).to_numpy(float)


def validate(sales: pd.DataFrame, start: int, end: int) -> dict:
    """Train before `start`, test on weeks [start, end) and report mean absolute error."""
    train = sales[sales["week"] < start]
    test = sales[(sales["week"] >= start) & (sales["week"] < end)]
    actual = test["units"].to_numpy(float)
    model = fit_demand_model(train)
    return {
        "mae_model": float(np.abs(predict_units(model, test) - actual).mean()),
        "mae_naive": float(np.abs(naive_forecast(train, test) - actual).mean()),
    }
