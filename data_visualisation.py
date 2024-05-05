import yfinance as yf
import mplfinance as mpf
import warnings
warnings.filterwarnings('ignore')

# Import data
data = yf.download("AAPL", start="2022-01-01", end="2022-06-30")

# Plot OHLC chart
mpf.plot(data)
# Plot candlestick chart
mpf.plot(data, type="candle")
# Plot line chart
mpf.plot(data, type="line")
# Plot Renko chart
mpf.plot(data, type="renko")
# Plot OHLC chart with 15-day moving average
mpf.plot(data, type="ohlc", mav=15)
# Include 3 moving averages
mpf.plot(data, type="candle", mav=(7, 14, 21))
# Include volume
mpf.plot(data, type="candle", mav=(7, 14, 21), volume=True)
# Include non-trading days
mpf.plot(
    data, 
    type="candle", 
    mav=(7, 14, 21), 
    volume=True, 
    show_nontrading=True
)

# Intraday charting
intraday = yf.download(tickers="PLTR", period="5d", interval="1m")
iday = intraday.iloc[-100:, :]
mpf.plot(iday, type="candle", mav=(7, 12), volume=True)

