import pandas as pd
import numpy as np
import yfinance as yf

tlt = yf.download("TLT", start="2002-01-01", end="2022-06-30")
tlt["log_return"] = np.log(tlt["Adj Close"] / tlt["Adj Close"].shift(1))
tlt["day_of_month"] = tlt.index.day
tlt["year"] = tlt.index.year
grouped_by_day = tlt.groupby("day_of_month").log_return.mean()
grouped_by_day.plot.bar()

tlt["first_week_returns"] = 0.0
tlt.loc[tlt.day_of_month <= 7, "first_week_returns"] = tlt[tlt.day_of_month <= 7].log_return

tlt["last_week_returns"] = 0.0
tlt.loc[tlt.day_of_month >= 23, "last_week_returns"] = tlt[tlt.day_of_month >= 23].log_return

tlt["last_week_less_first_week"] = tlt.last_week_returns - tlt.first_week_returns

(
    tlt.groupby("year")
    .last_week_less_first_week.mean()
    .plot.bar()
)

(
    tlt.groupby("year")
    .last_week_less_first_week.sum()
    .cumsum()
    .plot()
)

tlt.last_week_less_first_week.cumsum().plot()