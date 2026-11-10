"""Charts for the case study: forecast accuracy, A/B distributions, marketing ROI and price curve."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from predictive_decision.data import PRODUCTS, UNIT_COST, base_demand, expected_units  # noqa: E402
from predictive_decision.decision import PRICE_GRID  # noqa: E402


def save_figures(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _forecast_error(result, out_dir / "01_forecast_error.png"),
        _ab_distribution(result, out_dir / "02_ab_store_profit.png"),
        _roi_by_product(result, out_dir / "03_marketing_roi.png"),
        _price_curve(result, out_dir / "04_price_profit_curve.png"),
    ]


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _forecast_error(result, path):
    f = result["forecast"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Model (Poisson)", "Naive (last 4 weeks)"],
                  [f["mae_model"], f["mae_naive"]], color=["#2a6f97", "#b0b0b0"])
    ax.bar_label(bars, fmt="%.1f")
    ax.set_ylabel("MAE (units per product-week)")
    ax.set_title("Forecast error on holdout weeks 104-129")
    return _save(fig, path)


def _ab_distribution(result, path):
    ab = result["ab"]
    control, treatment = result["control"], result["treatment"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(control.min(), treatment.min()), max(control.max(), treatment.max()), 40)
    ax.hist(control, bins=bins, alpha=0.6, label="Current policy", color="#b0b0b0")
    ax.hist(treatment, bins=bins, alpha=0.6, label="New policy", color="#2a6f97")
    ax.axvline(control.mean(), color="#606060", linestyle="--")
    ax.axvline(treatment.mean(), color="#1d4e6b", linestyle="--")
    ax.set_xlabel("Profit per store over the 26-week horizon")
    ax.set_ylabel("Stores")
    ax.set_title(f"Simulated A/B: uplift {ab['uplift_pct']:.1f}% "
                 f"(95% CI {ab['ci_low_pct']:.1f}% to {ab['ci_high_pct']:.1f}%, p={ab['p_value']:.2f})")
    ax.legend()
    return _save(fig, path)


def _roi_by_product(result, path):
    roi = result["roi"].dropna().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(roi.index, roi.values, color="#2a6f97")
    ax.axvline(1.0, color="red", linestyle="--", label="Break-even (ROI = 1)")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xlabel("Incremental gross margin per currency unit of marketing")
    ax.set_title("Marketing ROI by product (chosen policy)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def _price_curve(result, path):
    product = "p02"
    i = PRODUCTS.index(product)
    d = result["policy"].loc[product]
    weeks = result["horizon"]
    base = base_demand(i)
    profits = [
        float((expected_units(base, p, d["marketing"], weeks) * (p - UNIT_COST) - d["marketing"]).sum())
        for p in PRICE_GRID
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(PRICE_GRID, profits, marker="o", color="#2a6f97")
    ax.axvline(d["price"], color="red", linestyle="--", label=f"Chosen price {d['price']:.0f}")
    ax.set_xlabel("Price")
    ax.set_ylabel("Expected profit over horizon")
    ax.set_title(f"Price vs profit for {product} at marketing {d['marketing']:.0f}")
    ax.legend()
    return _save(fig, path)
