

# Robust Portfolio Optimization using ARIMA, GARCH, CCM and Robust Optimization

This project implements a Robust Portfolio Optimization framework inspired by the research paper:

**“A Practical Guide to Robust Portfolio Optimization”**
by C. Yin, R. Perchet & F. Soupé 

The objective is to construct a stable and risk-aware portfolio by improving covariance estimation, reducing optimization sensitivity, and stress-testing the portfolio under uncertain market conditions.

Rather than relying only on traditional Markowitz optimization, this model integrates:

* ARIMA-based expected return estimation
* GARCH(1,1) volatility forecasting
* Sector-based Correlation Matrix Modeling (CCM)
* Robust Optimization with uncertainty penalty
* Condition Number analysis for covariance stability
* Out-of-sample testing
* Crash scenario robustness evaluation

---

# Project Motivation

Classical Mean-Variance Optimization is highly sensitive to estimation errors, especially in expected returns (μ). Even small forecasting errors can lead to unstable and unrealistic portfolio weights.

This project focuses on improving portfolio robustness by:

* stabilizing covariance structure
* modeling sector relationships explicitly
* penalizing uncertainty in optimization
* testing portfolio behavior under stressed market conditions

The goal is not only high return, but also portfolio stability and reliability.

---

# Methodology

## 1. Data Preparation

Historical stock price data is collected for multiple assets across different sectors:

* Energy / Utilities
* Banking / Financials
* Pharma / FMCG

Log returns are computed using:

[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
]

The dataset is divided into:

* 70% Training Data
* 30% Testing Data

for proper out-of-sample validation.

---

## 2. Expected Return Estimation (Current Approach)

Expected returns are currently estimated using:

## ARIMA (AutoRegressive Integrated Moving Average)

For each asset:

* AIC-based grid search selects the best ARIMA(p,0,q)
* One-step forecast is used as expected return

This provides the vector:

[
\mu = [\mu_1, \mu_2, ..., \mu_n]
]

---

## 3. Volatility Forecasting

Volatility is estimated using:

## GARCH(1,1)

This captures:

* volatility clustering
* heteroskedasticity
* time-varying market risk

which is significantly more realistic than using simple historical standard deviation.

---

## 4. Correlation Matrix Construction (CCM)

Instead of using raw sample correlation directly, the model builds a structured correlation matrix:

### Intra-sector correlation:

fixed using domain assumptions

### Cross-sector correlation:

optimized using Frobenius norm minimization

This improves covariance stability and reduces noise in estimation.

The covariance matrix becomes:

[
\Sigma = D \cdot C \cdot D
]

where:

* D = volatility matrix
* C = correlation matrix

---

## 5. Robust Optimization

Portfolio weights are obtained using robust optimization:

[
\max_w \left(
\mu^T w

* \lambda w^T \Sigma w
* k \sqrt{w^T \Omega w}
  \right)
  ]

where:

* Ω represents uncertainty matrix
* k controls robustness penalty

This reduces sensitivity to estimation errors and improves allocation stability.

---

## 6. Condition Number Analysis

Three condition numbers are calculated:

* Condition Number 1 → λ₁ / λₙ
* Condition Number 2 → λ₁ / λₙ₋₁
* Condition Number 3 → λ₁ / λₙ₋₂

These help evaluate:

* covariance matrix stability
* near-singularity issues
* robustness of optimization inputs

---

## 7. Out-of-Sample Testing

The optimized portfolio is evaluated on unseen test data using:

* Mean Return
* Annualized Volatility
* Sharpe Ratio

This ensures the model performs beyond the training sample.

---

# Key Observation

The most important finding from this project is:

## Mean estimation error dominates portfolio optimization

ARIMA-based expected return forecasts often showed inconsistencies with realized stock performance (measured using Sharpe ratios).

This means:

* assets with poor realized performance sometimes received large positive weights
* optimization became highly sensitive to noisy return forecasts

This validates a well-known result in portfolio theory:

> Expected return estimation is much harder than volatility and covariance estimation.

---

# Current Limitation

## ARIMA-based Expected Returns Need Improvement

While ARIMA helps generate expected returns, it has important limitations:

* assumes linear time-series behavior
* sensitive to short-term market noise
* weak under regime shifts
* may produce unrealistic directional forecasts

In several cases, ARIMA predictions did not align with realized Sharpe ratios, leading to unstable allocations.

This is currently the weakest part of the model.

---

# Future Improvement

## Black-Litterman Framework (Major Upgrade)

The next major improvement planned for this project is replacing raw ARIMA forecasts with:

# Black-Litterman Expected Return Estimation

Why?

Because Black-Litterman:

* reduces mean estimation error
* combines market equilibrium with investor views
* produces more stable expected returns
* avoids extreme portfolio weights
* improves economic interpretability

This would significantly improve portfolio quality and move the model closer to institutional-grade portfolio construction.

Instead of:

```text
Pure ARIMA Forecasts
```

the model will evolve toward:

```text
Black-Litterman + Robust Optimization + GARCH + CCM
```

which is far more reliable for real-world portfolio management.

---

# Additional Future Improvements

* Factor-based expected return models
* Transaction cost modeling
* Liquidity constraints

---

# Final Conclusion

This project demonstrates that:

## Robust covariance estimation is often more valuable than aggressive return forecasting

By focusing on:

* GARCH volatility
* structured correlations
* robust optimization
* stress testing

the portfolio becomes more stable and realistic.

The strongest lesson from this work is:

> Bad expected returns create bad portfolios.

Future development using Black-Litterman will address this directly and significantly improve the overall model quality.
