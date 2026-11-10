# predictive_decision_making_ML

Machine learning for business decisions: forecasting demand, comparing pricing and marketing options, and validating results with controlled experiments.

## Scope

This repository is a portfolio and working space for predictive decision-making in retail and e-commerce. The goal is to turn forecasts into decisions that can be measured:

- **Demand and revenue forecasting** at the product or store level.
- **Decision comparison**: estimate the expected profit and revenue of alternative prices, promotions, or order quantities.
- **Experiment validation**: confirm uplift with A/B tests before scaling a decision.
- **Business metrics**: profit, revenue, conversion, return on investment, and funnel analysis.

## Structure

```
predictive_decision_making_ML/
├── README.md
├── src/predictive_decision/
│   ├── data.py        # synthetic data generator (3 regions)
│   ├── model.py       # per-region linear regression and validation metrics
│   ├── economics.py   # break-even, top-N profit, bootstrap risk
│   └── __main__.py    # end-to-end decision summary
└── tests/             # pytest tests for each step
```

## Method

1. Train one linear regression per region with a 75/25 train/validation split.
2. Compute RMSE and the mean predicted volume per region.
3. Compare the predicted volume with the break-even volume (budget / revenue per unit).
4. Select the top 200 sites by predicted volume and compute their expected profit.
5. Estimate profit uncertainty with 1,000 bootstrap resamples (95% confidence interval and probability of loss).

## Run

```bash
pip install numpy pandas scikit-learn pytest
set PYTHONPATH=src          # PowerShell: $env:PYTHONPATH="src"
python -m pytest -q tests
python -m predictive_decision
```

## Results (synthetic data, seed 0)

| region | RMSE | break-even volume | profit (top 200) | 95% CI | loss risk |
|--------|------|-------------------|------------------|--------|-----------|
| region_a | 2.95 | 111.1 | -3.74M | [-4.62M, -2.95M] | 1.000 |
| region_b | 4.01 | 111.1 | -1.78M | [-2.60M, -0.91M] | 1.000 |
| region_c | 5.07 | 111.1 | 0.65M | [-0.42M, 1.54M] | 0.123 |

Only region_c has a positive expected profit, and even there the risk of loss is about 12%. The result shows why a forecast alone is not enough: the decision must account for uncertainty and the business break-even.

## Data policy

All data is synthetic and generated in code. No confidential or personal data is included.
