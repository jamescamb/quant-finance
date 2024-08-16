import numpy as np
import matplotlib.pyplot as plt

def simulate_trade(win_prob, avg_win, avg_loss):
    """
    Simulate a single trade with given win probability and average win/loss amounts.
    """
    if np.random.rand() < win_prob:
        return avg_win
    else:
        return -avg_loss
    
def simulate_trading_strategy(initial_capital, trades, win_prob, avg_win, avg_loss):
    """
    Simulate the entire trading strategy over a given number of trades.
    """
    capital = initial_capital
    capital_history = [capital]

    for _ in range(trades):
        capital += simulate_trade(win_prob, avg_win, avg_loss)
        capital_history.append(capital)

    return capital_history

def calculate_risk_of_ruin(initial_capital, trades, win_prob, avg_win, avg_loss, simulations=100):
    """
    Calculate the risk of ruin over a number of trading simulations.
    """
    ruin_count = 0

    for _ in range(simulations):
        capital_history = simulate_trading_strategy(initial_capital, trades, win_prob, avg_win, avg_loss)
        if min(capital_history) <= 0:
            ruin_count += 1

    return ruin_count / simulations

initial_capital = 10000
average_win = 110
average_loss = 100
trades = 1000

risk_of_ruins = []
steps = range(30, 60)
for step in steps:
    win_probability = step / 100
    risk_of_ruin = calculate_risk_of_ruin(initial_capital, trades, win_probability, average_win, average_loss)
    risk_of_ruins.append(risk_of_ruin)

# Plot the capital history
plt.figure(figsize=(10, 6))
plt.plot(steps, risk_of_ruins, label='Risk of ruin')
plt.xlabel('Probability of a winning trade')
plt.ylabel('Risk of ruin')
plt.grid(True)
plt.show()