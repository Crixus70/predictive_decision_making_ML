# Case study: pricing and marketing decisions for a retail portfolio

## Business problem

A retailer sells 20 products at a fixed price of 60 with a marketing budget of 100 per week per product. Management wants to know whether changing price and marketing spend would increase profit, and by how much, before rolling out any change.

## Approach

1. **Forecast demand** with a Poisson regression that uses price, marketing spend (with diminishing returns), product, and seasonality.
2. **Validate the forecast** on out-of-sample weeks against a naive baseline.
3. **Optimize** price and marketing per product to maximize expected gross margin minus marketing cost over a 26-week horizon.
4. **Test the decision** with a simulated store-level A/B experiment.

## Results

| Metric | Value |
|---|---|
| Forecast error (MAE, units/week) | 10.5 model vs 53.9 naive (-81%) |
| Expected profit, current policy | 1,225,272 |
| Expected profit, recommended policy | 1,289,928 (**+5.3%**) |
| A/B uplift (200 stores, simulated) | +3.8%, 95% CI -2.1% to +9.8%, p = 0.21 |
| Stores per arm needed to detect a 5% uplift | 549 |

### Recommended policy

- **Price: 66 for all products.** Profit peaks near this level in the price curve (figure 4).
- **Marketing: set per product** between 50 and 350 per week, following ROI.

### Marketing ROI by product

Marketing ROI is the incremental gross margin generated per currency unit of marketing spend. Values range from 1.40 to 2.94, so every product returns more than its marketing cost under the recommended policy (figure 3).

- Highest ROI: p02 (2.94), p08 (2.89), p09 (2.81).
- Lowest ROI: p14 (1.40), p01 (1.45), p07 (1.64).

## Conclusions

1. A demand model with diminishing marketing returns is needed. A linear model over-spent to the grid limit.
2. Setting price near the profit peak adds about 5% to expected profit.
3. The A/B result is not statistically significant at 200 stores. The test is underpowered; 549 stores per arm are needed to detect a 5% uplift.
4. Marketing budget should move toward products with higher ROI.

## Recommendations

1. Pilot the price of 66 in a subset of stores.
2. Reallocate marketing budget by product ROI, starting with the lowest-ROI products.
3. Run a larger experiment (at least 549 stores per arm) before scaling.
4. Track realized profit and ROI monthly, and refit the model with new data.

## Limitations

- **Synthetic data.** The generator uses the same functional form as the model, so forecast accuracy is optimistic. Real data needs holdout by store and time.
- **Oracle evaluation.** Decisions are scored against the simulator's true expected demand. Real profit will be lower and noisier.
- **Fixed cost assumption.** Unit cost is fixed at 40. Real cost and promotion effects must be modeled.

## Figures

- `figures/01_forecast_error.png`: model vs naive forecast error.
- `figures/02_ab_store_profit.png`: store profit distribution in the simulated A/B test.
- `figures/03_marketing_roi.png`: marketing ROI by product.
- `figures/04_price_profit_curve.png`: price vs expected profit for one product.
