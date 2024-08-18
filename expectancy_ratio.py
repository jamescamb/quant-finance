import pandas as pd
import numpy as np

def calculate_expectancy_ratio(trades):
    num_trades = len(trades)
    winners = trades[trades['Profit'] > 0]
    losers = trades[trades['Profit'] <= 0]

    win_rate = len(winners) / num_trades
    loss_rate = len(losers) / num_trades

    avg_win = winners['Profit'].mean()
    avg_loss = losers['Profit'].mean()

    expectancy_ratio = (win_rate * avg_win) + (loss_rate * avg_loss)

    return expectancy_ratio

trade_data = {
    'Trade': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Profit': [100, -50, 200, -100, 300, -150, 400, -200, 500, -250]
}

trades = pd.DataFrame(trade_data)

expectancy_ratio = calculate_expectancy_ratio(trades)
print(f"Expectancy Ratio: {expectancy_ratio}")