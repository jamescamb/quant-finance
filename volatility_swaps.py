import QuantLib as ql
import numpy as np
import pandas as pd


notional = 100_000
volatility_strike = 0.2438
days_to_maturity = 148
observation_period = 252

risk_free_rate = 0.0525
dividend_yield = 0.0052
spot_price = 188.64

calendar = ql.NullCalendar()
day_count = ql.Actual360()

today = ql.Date().todaysDate()
ql.Settings.instance().evaluationDate = today

risk_free_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(today, risk_free_rate, day_count)
)

dividend_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(today, dividend_yield, day_count)
)

# Underlying asset price
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))

strike_price = 190
option_price = 11.05
expiration_date = today + ql.Period(days_to_maturity, ql.Days)

payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
exercise = ql.EuropeanExercise(expiration_date)
european_option = ql.VanillaOption(payoff, exercise)

volatility_handle = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today, calendar, volatility_strike, day_count)
)

bsm_process = ql.BlackScholesMertonProcess(
    spot_handle, dividend_ts, risk_free_ts, volatility_handle
)

implied_volatility = european_option.impliedVolatility(
    option_price, bsm_process, 1e-4, 1000, 1e-8, 4.0
)

np.random.seed(42)

time_steps = observation_period
dt = 1 / observation_period

prices = np.zeros((time_steps + 1, 1))
prices[0] = spot_price

for t in range(1, time_steps + 1):
    z = np.random.normal(size=1)
    prices[t] = (
        prices[t-1] 
        * np.exp(
            (risk_free_rate - 0.5 * implied_volatility**2) * 
            dt + 
            implied_volatility * 
            np.sqrt(dt) * z
        )
    )

prices_df = pd.DataFrame(prices, columns=['Price'])

prices_df['Return'] = prices_df['Price'].pct_change().dropna()

realized_volatility = np.std(prices_df['Return']) * np.sqrt(observation_period)

time_to_maturity = days_to_maturity / observation_period

volatility_swap_value = (
    (realized_volatility - volatility_strike) * 
    notional * 
    np.sqrt(time_to_maturity)
)

print(f"Volatility Swap Value: ${volatility_swap_value:.2f}")