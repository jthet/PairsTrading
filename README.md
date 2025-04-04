# Systematic Pairs Trading

STAT 686: Market Models final project

Jackson Thetford, Ryker Dolese, Krish Kumar, Katharine Britt, Naomi Consiglio Mehrdad Tamiji


## PairsFinder Class Documentation

### Overview
The `PairsFinder` class is designed to streamline the process of identifying cointegrated pairs for pairs trading strategies. It automates several key steps: data loading, data splitting, return preprocessing, autoencoder training for dimensionality reduction, asset clustering, and cointegration analysis combined with a Hurst exponent test to select mean-reverting pairs.

### Key Features

- **Data Loading:**  
  Automatically downloads historical price data using yfinance if a local CSV file does not exist.

- **Data Splitting:**  
  Splits the loaded data into training and testing sets based on a customizable split ratio.

- **Preprocessing:**  
  Computes percentage returns and applies standard scaling while filtering out assets with no variation.

- **Autoencoder Training:**  
  Reduces the dimensionality of the asset returns using a customizable autoencoder, with options to adjust:
  - Encoding dimension
  - Number of epochs
  - Learning rate
  - Hidden layer size

- **Clustering:**  
  Clusters assets using KMeans on the encoded returns.  
  *Note:* If the number of clusters is not specified, it defaults to approximately one-fourth of the number of assets.

- **Cointegration Analysis:**  
  Evaluates asset pairs for cointegration (using a p-value threshold) and tests for mean reversion by computing the Hurst exponent of the spread.

- **Pipeline Execution:**  
  The `run_pipeline()` method ties all steps together for a seamless end-to-end analysis.

- **Logging:**  
  Uses Python’s logging module to provide status updates and debug information throughout the process.

### Usage Example

```python
import pandas as pd
import logging
from pairsfinder import PairsFinder  # Adjust import based on your module location

# Load tickers (assumes 'russel3000_stocks.csv' exists)
tickers = pd.read_csv('russel3000_stocks.csv')['Ticker'].tolist()

# Optional: Customize autoencoder and clustering parameters
autoencoder_params = {
    'encoding_dim': 10,     # 10-dimensional latent space
    'num_epochs': 500,      # Train for 500 epochs
    'lr': 0.005,            # Learning rate of 0.005
    'print_every': 25,      # Log every 25 epochs
    'hidden_dim': 128       # Hidden layer size of 128 neurons
}

clustering_params = {
    'num_clusters': 100,    # Set number of clusters to 100
    'random_state': 123     # Specific random state for reproducibility
}

# Other custom settings
cointegration_threshold = 0.05
hurst_threshold = 0.45
min_cluster_size = 4
split_ratio = 0.6

# Instantiate the PairsFinder class with custom parameters
pf = PairsFinder(
    tickers=tickers,
    start="2019-01-01", 
    end="2025-01-01",
    autoencoder_params=autoencoder_params,
    clustering_params=clustering_params,
    cointegration_threshold=cointegration_threshold,
    hurst_threshold=hurst_threshold,
    min_cluster_size=min_cluster_size,
    split_ratio=split_ratio,
    log_level=logging.INFO  # Change to logging.DEBUG for more details
)

# Run the complete pipeline and save the resulting pairs to a CSV file
pairs_df = pf.run_pipeline()
pf.save_pairs("custom_pairs_df.csv")
