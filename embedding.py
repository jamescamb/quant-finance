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

"""
This type of network is useful for learning efficient representations (embeddings) of data,
which can be used for tasks such as dimensionality reduction, denoising, or anomaly detection.
"""

# Now let's train the autoencoder
def train(model, data_loader, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for data in data_loader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

feature_dim = processed_data.shape[1]
model = StockAutoencoder(feature_dim)
train(model, data_loader)

def extract_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0]
            encoded = model.encoder(inputs)
            embeddings.append(encoded)
    return torch.vstack(embeddings)

# Extract the embeddings and use them to create clusters
embeddings = extract_embeddings(model, data_loader)
kmeans = KMeans(n_clusters=5, random_state=42).fit(embeddings.numpy())
clusters = kmeans.labels_

"""
Principal Component Analysis (PCA) reduces the dimensionality of the embeddings to principal components.
These components capture the directions of maximum variance in the data.
"""

# Initialise a PCA model to reduce the dimensionality of the embeddings to two principal components
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings.numpy())

# Convert the embeddings into a two-dimensional format so we can plot them
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=clusters,
    palette=sns.color_palette("hsv", len(set(clusters))),
)
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()