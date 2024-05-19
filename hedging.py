"""
Hedging systematic sector risk and calculating the effective breadth
"""

# Import libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# Import financial data
tickers = ["WFC", "JPM", "USB", "XOM", "VLO", "SLB"]
data = yf.download(tickers, start="2015-01-01", end="2023-12-31")["Close"]
returns = data.pct_change().dropna()

# Plot close price and correlation matrix
fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.tight_layout()
corr = returns.corr()
left = data.plot(ax=ax1)
right = sns.heatmap(
    corr, ax=ax2, vmin=-1, vmax=1,
    xticklabels=tickers, yticklabels=tickers
)
plt.show()

market_symbols = ["XLF", "SPY", "XLE"]
sector_1_stocks = ["WFC", "JPM", "USB"]
sector_2_stocks = ["XOM", "VLO", "SLB"]

tickers = market_symbols + sector_1_stocks + sector_2_stocks
price = yf.download(tickers, start="2015-01-01", end="2023-12-31").Close
returns = price.pct_change().dropna()

market_returns = returns["SPY"]
sector_1_returns = returns["XLF"]
sector_2_returns = returns["XLE"]

stock_returns = returns.drop(market_symbols, axis=1)
residuals_market = stock_returns.copy() * 0.0
residuals = stock_returns.copy() * 0.0

def ols_residual(y, x):
    results = sm.OLS(y, x).fit()
    return results.resid

sector_1_excess = ols_residual(sector_1_returns, market_returns)
sector_2_excess = ols_residual(sector_2_returns, market_returns)

for stock in sector_1_stocks:
    residuals_market[stock] = ols_residual(returns[stock], market_returns)
    residuals[stock] = ols_residual(residuals_market[stock], sector_1_excess)

for stock in sector_2_stocks:
    residuals_market[stock] = ols_residual(returns[stock], market_returns)
    residuals[stock] = ols_residual(residuals_market[stock], sector_2_excess)

fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.tight_layout()
corr = residuals.corr()
left = (1 + residuals).cumprod().plot(ax=ax1)
right = sns.heatmap(
    corr,
    ax=ax2,
    fmt="d",
    vmin=-1,
    vmax=1,
    xticklabels=residuals.columns,
    yticklabels=residuals.columns,
)
plt.show()

# Calculate effective breadth
def buckle_BR_const(N, rho):
    return N/(1 + rho*(N - 1))

corr = np.linspace(start=0, stop=1.0, num=500)
plt.plot(corr, buckle_BR_const(6, corr))
plt.ylabel('Effective Breadth (Number of Bets)')
plt.xlabel('Forecast Correlation')
plt.show()