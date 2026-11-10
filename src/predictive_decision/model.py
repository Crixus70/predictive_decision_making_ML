"""Train a linear model per region and report validation metrics."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

FEATURES = ["f0", "f1", "f2"]


def train_region(df: pd.DataFrame, seed: int = 0) -> dict:
    """Split 75:25, fit a linear regression and return the model with validation results."""
    train, valid = train_test_split(df, test_size=0.25, random_state=seed)
    model = LinearRegression().fit(train[FEATURES], train["product"])
    preds = pd.Series(model.predict(valid[FEATURES]), index=valid.index, name="predicted")
    return {
        "model": model,
        "validation": pd.DataFrame({"actual": valid["product"], "predicted": preds}),
        "rmse": float(np.sqrt(mean_squared_error(valid["product"], preds))),
        "mean_predicted": float(preds.mean()),
    }
