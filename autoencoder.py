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

# Import stock data
symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "BRK-B", "V", "JNJ", "WMT", "JPM",
    "MA", "PG", "UNH", "DIS", "NVDA", "HD", 
    "PYPL", "BAC", "VZ", "ADBE", "CMCSA", "NFLX",
    "KO", "NKE", "MRK", "PEP", "T", "PFE", "INTC",
]
symbols = ['MMM','AOS','ABT','ABBV','ACN','ADBE','AMD','AES','AFL','A','APD','AKAM','ALB','ARE',
		   'ALGN','ALLE','LNT','ALL','GOOGL','GOOG','MO','AMZN','AMCR','AEE','AAL','AEP','AXP',
		   'AIG','AMT','AWK','AMP','AME','AMGN','APH','ADI','ANSS','AON','APA','AAPL','AMAT','APTV',
		   'ACGL','ADM','ANET','AJG','AIZ','T','ATO','ADSK','ADP','AZO','AVB','AVY','AXON','BKR','BALL',
		   'BAC','BK','BBWI','BAX','BDX','BBY','BIO','TECH','BIIB','BLK','BX','BA','BKNG','BWA','BXP',
		   'BSX','BMY','AVGO','BR','BRO','BLDR','BG','CDNS','CZR','CPT','CPB','COF','CAH','KMX','CCL',
		   'CTLT','CAT','CBOE','CBRE','CDW','CE','COR','CNC','CNP','CF','CHRW','CRL','SCHW','CHTR','CVX',
		   'CMG','CB','CHD','CI','CINF','CTAS','CSCO','C','CFG','CLX','CME','CMS','KO','CTSH','CL','CMCSA',
		   'CMA','CAG','COP','ED','STZ','COO','CPRT','GLW','CTVA','CSGP','COST','CTRA','CCI','CSX','CMI',
		   'CVS','DHR','DRI','DVA','DAY','DE','DAL','XRAY','DVN','DXCM','FANG','DLR','DFS','DG','DLTR'
		   ,'D','DPZ','DOV','DOW','DHI','DTE','DUK','DD','EMN','ETN','EBAY','ECL','EIX','EW','EA','ELV',
		   'LLY','EMR','ENPH','ETR','EOG','EPAM','EQT','EFX','EQIX','EQR','ESS','EL','ETSY','EG','EVRG',
		   'ES','EXC','EXPE','EXPD','EXR','XOM','FFIV','FDS','FICO','FAST','FRT','FDX','FIS','FITB','FSLR',
		   'FE','FI','FLT','FMC','F','FTNT','FTV','FOXA','FOX','BEN','FCX','GRMN','IT','GEN','GNRC','GD',
		   'GE','GIS','GM','GPC','GILD','GPN','GL','GS','HAL','HIG','HAS','HCA','HSIC','HSY','HES',
		   'HPE','HLT','HOLX','HD','HON','HRL','HST','HWM','HPQ','HUBB','HUM','HBAN','HII','IBM','IEX',
		   'IDXX','ITW','ILMN','INCY','IR','PODD','INTC','ICE','IFF','IP','IPG','INTU','ISRG','IVZ','INVH',
		   'IQV','IRM','JBHT','JBL','JKHY','J','JNJ','JCI','JPM','JNPR','K','KDP','KEY','KEYS','KMB','KIM',
		   'KMI','KLAC','KHC','KR','LHX','LH','LRCX','LW','LVS','LDOS','LEN','LIN','LYV','LKQ','LMT','L',
		   'LOW','LULU','LYB','MTB','MRO','MPC','MKTX','MAR','MMC','MLM','MAS','MA','MTCH','MKC','MCD','MCK',
		   'MDT','MRK','META','MET','MTD','MGM','MCHP','MU','MSFT','MAA','MRNA','MHK','MOH','TAP','MDLZ',
		   'MPWR','MNST','MCO','MS','MOS','MSI','MSCI','NDAQ','NTAP','NFLX','NEM','NWSA','NWS','NEE','NKE',
		   'NI','NDSN','NSC','NTRS','NOC','NCLH','NRG','NUE','NVDA','NVR','NXPI','ORLY','OXY','ODFL','OMC',
		   'ON','OKE','ORCL','PCAR','PKG','PANW','PARA','PH','PAYX','PAYC','PYPL','PNR','PEP','PFE','PCG',
		   'PM','PSX','PNW','PXD','PNC','POOL','PPG','PPL','PFG','PG','PGR','PLD','PRU','PEG','PTC','PSA',
		   'PHM','QRVO','PWR','QCOM','DGX','RL','RJF','RTX','O','REG','REGN','RF','RSG','RMD','RVTY','RHI',
		   'ROK','ROL','ROP','ROST','RCL','SPGI','CRM','SBAC','SLB','STX','SRE','NOW','SHW','SPG','SWKS','SJM',
		   'SNA','SO','LUV','SWK','SBUX','STT','STLD','STE','SYK','SYF','SNPS','SYY','TMUS','TROW','TTWO','TPR',
		   'TRGP','TGT','TEL','TDY','TFX','TER','TSLA','TXN','TXT','TMO','TJX','TSCO','TT','TDG','TRV','TRMB',
		   'TFC','TYL','TSN','USB','UBER','UDR','ULTA','UNP','UAL','UPS','URI','UNH','UHS','VLO','VTR','VRSN',
		   'VRSK','VZ','VRTX','VFC','VTRS','VICI','V','VMC','WRB','WAB','WBA','WMT','DIS','WBD','WM','WAT','WEC',
		   'WFC','WELL','WST','WDC','WRK','WY','WHR','WMB','WTW','GWW','WYNN','XEL','XYL','YUM','ZBRA','ZBH','ZION','ZTS']
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
x_train, x_test = stock_data[0 : len(symbols) - 5], stock_data[len(symbols) - 5 : len(symbols)]

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
			layers.Dense(500, activation='relu'),
			layers.Dense(100, activation='relu'),
			layers.Dense(latent_dimensions, activation='relu'),
		])

		# Decoder architecture using another Sequential model
		self.decoder = tf.keras.Sequential([
			layers.Dense(500, activation='relu'),
			layers.Dense(100, activation='relu'),
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
latent_dimensions = 5

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