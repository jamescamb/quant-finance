"""
Calculate excess returns of the portfolio over the benchmark,
and compute the tracking error
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# Generate sample data
symbols = [
    "XLE",
    "XLF",
    "XLU",
    "XLI",
    "GDX",
    "XLK",
    "XLV",
    "XLY",
    "XLP",
    "XLB",
    "SPY"
]
data = yf.download(symbols, start="2020-01-01")["Adj Close"]
returns = data.pct_change().dropna()
benchmark_returns = returns.pop("SPY")
portfolio_returns = returns.sum(axis=1)

excess_returns = portfolio_returns - benchmark_returns

tracking_error = excess_returns.std()

plt.figure(figsize=(14, 7))
plt.plot(portfolio_returns.index, portfolio_returns, label='Portfolio Returns', lw=0.5)
plt.plot(benchmark_returns.index, benchmark_returns, label='Benchmark Returns', lw=0.5)
plt.plot(excess_returns.index, excess_returns, label='Excess Returns', linestyle='--', lw=0.5)
plt.legend()
plt.title('Portfolio vs Benchmark Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.show()