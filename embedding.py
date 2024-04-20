"""
Code to train an autoencoder to build embeddings for stock factors
"""

# Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import stock data
symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "BRK-B", "V", "JNJ", "WMT", "JPM",
    "MA", "PG", "UNH", "DIS", "NVDA", "HD", 
    "PYPL", "BAC", "VZ", "ADBE", "CMCSA", "NFLX",
    "KO", "NKE", "MRK", "PEP", "T", "PFE", "INTC",
]
stock_data = yf.download(
    symbols, 
    start="2020-01-01", 
    end="2023-12-31"
)["Adj Close"]

# Create some features
log_returns = np.log(stock_data / stock_data.shift(1))
moving_avg = stock_data.rolling(window=22).mean()
volatility = stock_data.rolling(window=22).std()

features = pd.concat([log_returns, moving_avg, volatility], axis=1).dropna()
processed_data = (features - features.mean()) / features.std()

# Build an autoencoder
tensor = torch.tensor(processed_data.values, dtype=torch.float32)
dataset = TensorDataset(tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

"""
An autoencoder is a type of neural network that learns to
compress (encode) the input data into a smaller representation
and then reconstruct (decode) the output to match the input as closely as possible.
"""

class StockAutoencoder(nn.Module):
    def __init__(self, feature_dim):
        super(StockAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),  # Latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

