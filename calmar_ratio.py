import empyrical as ep
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

symbols = [
    "AAPL",
    "AMZN",
    "GOOG",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
]

data = yf.download(symbols, start="2020-01-01")["Adj Close"]
returns = data.pct_change().dropna()
portfolio_returns = returns.sum(axis=1)

# Compute the Calmar Ratio using Empyrical
calmar_ratio = ep.calmar_ratio(portfolio_returns)
print(f"Calmar Ratio: {calmar_ratio}")