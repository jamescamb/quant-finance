import pandas as pd
import numpy as np
import yfinance as yf

ticker = "SPY"
data = yf.download(ticker)

# Calculate daily returns
data["return"] = data["Close"].diff()

# Identify days with negative returns
data["down"] = data["return"] < 0

# Identify 3-day losing streaks
data["3_day_losing_streak"] = (data["down"] & data["down"].shift(1) & data["down"].shift(2))

# Initialize a column to keep track of days since last 3-day losing streak
data["days_since_last_streak"] = np.nan

# Iterate over the data to calculate the days since the last streak
last_streak_day = -np.inf  # Initialize with a very large negative value

for i in range(len(data)):
    if data["3_day_losing_streak"].iloc[i]:
        if i - last_streak_day >= 42:  # Check if it's been at least 42 trading days
            data.loc[data.index[i], "days_since_last_streak"] = i - last_streak_day
        last_streak_day = i

# Filter the data to show only the occurrences that meet the criteria
result = data.dropna(subset=["days_since_last_streak"]).copy()

# Calculate future returns
result["next_1_day_return"] = data["Close"].shift(-1) / data["Close"] - 1
result["next_5_day_return"] = data["Close"].shift(-5) / data["Close"] - 1
result["next_10_day_return"] = data["Close"].shift(-10) / data["Close"] - 1
result["next_21_day_return"] = data["Close"].shift(-21) / data["Close"] - 1

# Print the results
cols = ["next_1_day_return", 
        "next_5_day_return", 
        "next_10_day_return",
        "next_21_day_return"]
print(result[cols].mean())
result[cols].gt(0).mean().plot.bar()