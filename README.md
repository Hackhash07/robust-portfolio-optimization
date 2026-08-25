Robust Portfolio Optimization using ARIMA, GARCH, CCM and Robust Optimization

This project is about building a portfolio that is not only designed to earn good returns, but is also able to remain stable when the estimates used by the model are wrong.

Traditional portfolio optimization can be very sensitive to small changes in expected returns and risk estimates. Because of this, even a small forecasting error can sometimes lead to very different portfolio weights.

So, in this project, I combine several methods to make the portfolio more reliable.

What the project does

I start by collecting historical price data for stocks from different sectors. I convert the price data into daily returns and divide the data into two parts: one part is used to build the model and the other part is kept aside to test how well the model actually performs.

Expected returns using ARIMA

For each stock, I use ARIMA to forecast its expected return.

The idea is to look at the historical behavior of each stock and use that information to estimate what its return might look like in the near future.

I test different ARIMA models and select the one that performs best according to the model selection criteria.

These forecasts are then given to the portfolio optimizer.

Volatility using GARCH

Instead of assuming that a stock always has the same level of risk, I use GARCH to forecast volatility.

This is useful because financial markets usually experience periods of high volatility followed by high volatility, and calm periods followed by relatively calm periods.

GARCH helps the model capture this changing risk over time.

Correlation using CCM

The next part is the correlation structure.

Instead of directly trusting the historical correlations between all the stocks, I create a more structured correlation matrix based on the relationships between sectors.

For example, stocks operating in the same sector can be expected to behave more similarly, while stocks from very different sectors may have weaker relationships.

This helps reduce noise in the correlation estimates and makes the covariance matrix more stable.

Robust optimization

Once the expected returns, volatility and correlations are estimated, I use robust optimization to determine the portfolio weights.

The important difference compared with traditional optimization is that I do not assume that my estimates are perfectly accurate.

The model adds a penalty for uncertainty.

So, if a portfolio looks attractive only because of an uncertain estimate, the optimizer becomes more cautious about giving that portfolio a large weight.

The parameter controlling this penalty is called kappa.

A small kappa means the model behaves more like a normal optimizer.

A larger kappa means the model becomes more conservative and places more importance on uncertainty.

Condition number analysis

I also analyze the condition number of the covariance matrix.

The purpose is to check whether the covariance matrix is stable or whether it is close to becoming singular.

This matters because an unstable covariance matrix can make optimization extremely sensitive to small changes in the data.

So this part is basically a diagnostic step to understand how reliable the risk estimates are before using them in optimization.

Out-of-sample testing

After building the portfolio using the training data, I test it on completely unseen data.

I look at things like:

average return
volatility
Sharpe ratio

The important point is that the test data was not used while constructing the portfolio.

This gives a better idea of whether the strategy actually works or whether it only looked good on historical data.

Main finding

The biggest thing I have observed so far is that expected return estimation is the most problematic part of the model.

The ARIMA forecasts sometimes do not match the actual performance of the stocks afterward.

For example, a stock may receive a large allocation because the model predicts a strong future return, but the stock may later perform poorly.

This shows that even if the risk model is reasonably good, inaccurate expected return forecasts can still produce poor portfolio allocations.

That is probably the most important lesson from the project.

Current limitation

The main weakness of the current framework is ARIMA-based expected returns.

ARIMA works well for many traditional time-series problems, but financial markets are much more complicated.

Market behavior can change because of:

economic conditions
market regimes
unexpected news
investor sentiment
sudden volatility

Because of this, a short-term statistical forecast can sometimes be unreliable.

Planned improvement: Black-Litterman

The next major improvement is to introduce Black-Litterman for expected returns.

Instead of depending completely on ARIMA forecasts, Black-Litterman allows the model to combine a market-based expected return with specific views about individual assets.

The main advantage is that it should produce more stable expected returns and reduce the extreme allocations that can occur when the optimizer trusts noisy forecasts too much.

So the project will eventually move toward:

Black-Litterman for expected returns + GARCH for volatility + CCM for correlations + Robust Optimization for portfolio construction.

Final takeaway

The main lesson from this project is:

The optimizer itself is not necessarily the main problem. The quality of the inputs given to the optimizer is.

Even a mathematically well-designed optimization model can produce a bad portfolio if the expected returns are poor.

That is why my next focus is improving the expected return estimation rather than simply making the optimizer more complicated.
