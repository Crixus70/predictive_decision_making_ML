"""Synthetic regional site data with three features and a target volume (thousands of units)."""

import numpy as np
import pandas as pd

REGION_NAMES = ("region_a", "region_b", "region_c")


def generate_regions(n_sites: int = 10_000, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Return one DataFrame per region with features f0, f1, f2 and the target `product`."""
    rng = np.random.default_rng(seed)
    regions = {}
    for i, name in enumerate(REGION_NAMES):
        f0, f1, f2 = rng.normal(0, 1, (3, n_sites))
        noise = rng.normal(0, 3 + i, n_sites)
        product = 90 + 8 * f0 + 3 * f1 - 4 * f2 + 2 * i + noise
        regions[name] = pd.DataFrame({
            "f0": f0, "f1": f1, "f2": f2,
            "product": np.clip(product, 0, None),
        })
    return regions
