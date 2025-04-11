# Systematic Pairs Trading

STAT 686: Market Models final project

Jackson Thetford, Ryker Dolese, Krish Kumar, Katharine Britt, Naomi Consiglio Mehrdad Tamiji


## PairsFinder Class

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
```

## Strategy Classes



















Below is a complete, finished version of your README.md that covers the PairsFinder class, the strategy classes, usage examples, and the key assumptions made throughout the project.

---

# Systematic Pairs Trading

STAT 686: Market Models Final Project  
Jackson Thetford, Ryker Dolese, Krish Kumar, Katharine Britt, Naomi Consiglio, Mehrdad Tamiji

---

## Overview

This project implements a systematic pairs trading framework. The system involves two major components:  
1. **PairsFinder Class:** A module that automates the identification of cointegrated (i.e., mean-reverting) asset pairs from a universe of stocks using various statistical techniques and machine learning methods for dimensionality reduction and clustering.  
2. **Strategy Classes:** Modules that implement different trading strategies (e.g., Z‑Score, XGBoost, and Logistic Regression) on the identified pairs. These strategies simulate trade execution, manage positions, and calculate portfolio returns.

---

## PairsFinder Class

### Overview
The `PairsFinder` class streamlines the process of identifying cointegrated pairs for pairs trading strategies. It automates key steps: data loading, data splitting, return preprocessing, autoencoder training for dimensionality reduction, asset clustering, and cointegration analysis (including a Hurst exponent test) to select mean-reverting pairs.

### Key Features

- **Data Loading:**  
  Automatically downloads historical price data using yfinance if a local CSV file does not exist.

- **Data Splitting:**  
  Splits the loaded data into training and testing sets based on a customizable split ratio.

- **Preprocessing:**  
  Computes percentage returns and applies standard scaling while filtering out assets with no variation.

- **Autoencoder Training:**  
  Reduces the dimensionality of the asset returns using a customizable autoencoder. Parameters include:
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
```

---

## Strategy Classes

This project implements multiple strategies for executing a pairs trading approach on the cointegrated pairs identified by `PairsFinder`. Each strategy takes a different approach to generating trading signals. Total returns are the weighted average return for the trading backtest for each pair, where the weights are defined by the quality score, or the level of strength of cointegration.


### Z-Score Strategy

#### Overview
The Z‑Score strategy bases its trading decisions on the statistics of the spread between a pair of assets. It calculates a z‑score from the spread (computed on log‑transformed prices using an OLS hedge ratio) and then enters a trade when the z‑score exceeds a specified upper or lower threshold. Positions are closed when the spread reverts toward its mean or when risk management conditions (stop loss or take profit) are met.

#### Usage

```python
from ZScoreStrategy import ZScoreStrategy

pairs_df = pd.read_csv("data/cointegrated_pairs.csv")
test_data = pd.read_csv("data/russel_data_test.csv", index_col=0, parse_dates=True)
full_data = pd.read_csv("data/russel_data_full.csv", index_col=0, parse_dates=True)

zscore_strategy = ZScoreStrategy(pairs_df, test_data, full_data=full_data, capital=100)
zscore_strategy.run()

zscore_strategy.save_results("zscore_results.csv")
```

---

### XGBoost Strategy

#### Overview
The XGBoost strategy uses an ensemble of decision trees (via the XGBoost algorithm) trained on engineered features from the asset pair’s spread. It predicts the probability of a favorable future movement in the spread. Positions are entered based on a combination of z‑score thresholds and minimum prediction probability.

#### Usage

```python
from XGBoostStrategy import XGBoostStrategy

pairs_df = pd.read_csv("data/cointegrated_pairs.csv")
test_data = pd.read_csv("data/russel_data_test.csv", index_col=0, parse_dates=True)
full_data = pd.read_csv("data/russel_data_full.csv", index_col=0, parse_dates=True)

xgb_strategy = XGBoostStrategy(pairs_df, test_data, full_data=full_data, capital=100,
                               min_zscore_threshold=0.5, min_proba_threshold=0.6)
xgb_strategy.run()
xgb_strategy.save_results("xgboost_results.csv")
```

---

### Logistic Regression Strategy

#### Overview
The Logistic Regression strategy uses a linear classifier to estimate the probability that the spread will move in a favorable direction. It uses engineered features similar to the XGBoost model. Trade signals are generated based on the predicted probabilities exceeding a predefined threshold. 


#### Usage Example

```python
from LogisticStrategy import LogisticStrategy

pairs_df = pd.read_csv("data/cointegrated_pairs.csv")
test_data = pd.read_csv("data/russel_data_test.csv", index_col=0, parse_dates=True)
full_data = pd.read_csv("data/russel_data_full.csv", index_col=0, parse_dates=True)

logistic_strategy = LogisticStrategy(pairs_df, test_data, full_data=full_data, capital=100,
                                     min_zscore_threshold=0.5, min_proba_threshold=0.6)
logistic_strategy.run()
logistic_strategy.save_results("logistic_results.csv")
```

---

### Strategy Assumptions

- **Risk-Free Rate:**  
  - The portfolio performance metrics (e.g., Sharpe ratio) are calculated with the assumption that the risk‑free rate is zero.

- **Execution and Transaction Costs:**  
  - The simulation does not incorporate transaction costs, slippage, or bid/ask spread.
