import yfinance as yf
import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras import layers, losses
from tensorflow import keras
#from keras.datasets import mnist
from keras.models import Model

from pypfopt import risk_models
from pypfopt import expected_returns

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

stock_data = stock_data.dropna()

S = risk_models.sample_cov(stock_data)
mu = expected_returns.mean_historical_return(stock_data)
print(S.head(5))
print(mu.head(5))

stock_data = stock_data.values.T

# Definition of the Autoencoder model as a subclass of the TensorFlow Model class

class SimpleAutoencoder(Model):
	def __init__(self,latent_dimensions , data_shape):
		super(SimpleAutoencoder, self).__init__()
		self.latent_dimensions = latent_dimensions
		self.data_shape = data_shape

		# Encoder architecture using a Sequential model
		self.encoder = tf.keras.Sequential([
			#layers.Flatten(),
			layers.Dense(latent_dimensions, activation='relu'),
		])

		# Decoder architecture using another Sequential model
		self.decoder = tf.keras.Sequential([
			layers.Dense(tf.math.reduce_prod(data_shape), activation='sigmoid'),
			#layers.Reshape(data_shape)
		])

	# Forward pass method defining the encoding and decoding steps
	def call(self, input_data):
		encoded_data = self.encoder(input_data)
		decoded_data = self.decoder(encoded_data)
		return decoded_data

# Extracting shape information from the testing dataset
input_data_shape = stock_data.shape[1:]

# Specifying the dimensionality of the latent space
latent_dimensions = 10

# Creating an instance of the SimpleAutoencoder model
simple_autoencoder = SimpleAutoencoder(latent_dimensions, input_data_shape)

simple_autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

simple_autoencoder.fit(stock_data, stock_data,
				epochs=1,
				shuffle=True)
