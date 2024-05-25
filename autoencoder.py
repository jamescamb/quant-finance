"""
Program to implement an autoencoder for stock data
"""

# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Neural network libraries
import tensorflow as tf
from keras import layers, losses
from keras.models import Model

# Financial libraries
import yfinance as yf
from pypfopt import risk_models
from pypfopt import expected_returns

#tickers = pd.read_csv('tickers.csv')
#symbols = list(tickers.values.flatten())[100:164]
#print(symbols)

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

# Remove rows of missing data
stock_data = stock_data.dropna()

# Calculate annualised covariance matrix and returns
S = risk_models.sample_cov(stock_data)
mu = expected_returns.mean_historical_return(stock_data)
print(S.head(5))
print(mu.head(5))

# Split the data into training and testing sets
stock_data = stock_data.values.T
x_train, x_test = stock_data[0:24], stock_data[24:30]

# Normalise the data ot [0, 1]
x_train = x_train / np.max(x_train, axis=1)[:, None]
x_test = x_test / np.max(x_test, axis=1)[:, None]
print(x_test)

# Display the shapes of the training and testing datasets
print("Shape of the training data:", x_train.shape)
print("Shape of the testing data:", x_test.shape)

# Definition of the Autoencoder model as a subclass of the TensorFlow Model class
class SimpleAutoencoder(Model):

	def __init__(self, latent_dimensions, data_shape):
		
		super(SimpleAutoencoder, self).__init__()
		self.latent_dimensions = latent_dimensions
		self.data_shape = data_shape

		# Encoder architecture using a Sequential model
		self.encoder = tf.keras.Sequential([
			layers.Flatten(),
			layers.Dense(latent_dimensions, activation='relu'),
		])

		# Decoder architecture using another Sequential model
		self.decoder = tf.keras.Sequential([
			layers.Dense(tf.math.reduce_prod(data_shape).numpy(), activation='sigmoid'),
			layers.Reshape(data_shape)
		])

	# Forward pass method defining the encoding and decoding steps
	def call(self, input_data):
		encoded_data = self.encoder(input_data)
		decoded_data = self.decoder(encoded_data)
		return decoded_data

# Extracting shape information from the testing dataset
input_data_shape = x_test.shape[1:]
print(input_data_shape)

# Specifying the dimensionality of the latent space
latent_dimensions = 100

# Creating an instance of the SimpleAutoencoder model
simple_autoencoder = SimpleAutoencoder(latent_dimensions, input_data_shape)

simple_autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

simple_autoencoder.fit(x_train, x_train,
 				epochs=100,
 				shuffle=True,
 				validation_data=(x_test, x_test))

encoded_imgs = simple_autoencoder.encoder(x_test).numpy()
decoded_imgs = simple_autoencoder.decoder(encoded_imgs).numpy()

n = 5
plt.figure(figsize=(8, 4))
for i in range(n):
	ax = plt.subplot(2, n, i + 1)
	plt.plot(x_test[i])
	plt.title("original")
	#plt.gray()
	
	# display reconstruction
	ax = plt.subplot(2, n, i + 1 + n)
	plt.plot(decoded_imgs[i])
	plt.title("reconstructed")
	#plt.gray()
plt.show()