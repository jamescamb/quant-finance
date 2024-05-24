"""
Simulate stock prices using Geometric Brownian Motion (GBM)
"""

import numpy as np
import matplotlib.pyplot as plt

# Setup Brownian motion parameters
s0 = 131.00
sigma = 0.25
mu = 0.35

# Setup the simulation
paths = 1000
delta = 1.0/252.0
time = 252 * 5

def wiener_process(delta, sigma, time, paths):

    # Return an array of samples from a normal distribution
    return sigma * np.random.normal(loc = 0, scale = np.sqrt(delta), size = (time, paths))

def gbm_returns(delta, sigma, time, mu, paths):

    process = wiener_process(delta, sigma, time, paths)

    return np.exp(process + (mu - sigma**2 / 2) * delta)

def gbm_levels(s0, delta, sigma, time, mu, paths):

    returns = gbm_returns(delta, sigma, time, mu, paths)
    stacked = np.vstack([np.ones(paths), returns])

    return s0 * stacked.cumprod(axis=0)

# Simulation of Apple returns
price_paths = gbm_levels(s0, delta, sigma, time, mu, paths)
plt.plot(price_paths, linewidth=0.25)
plt.show()

# Simulation now with zero drift
price_paths = gbm_levels(s0, delta, sigma, time, 0.0, paths)
plt.plot(price_paths, linewidth=0.25)
plt.show()