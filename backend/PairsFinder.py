import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.stattools import coint
from hurst import compute_Hc

class PairsFinder():
    class Autoencoder(nn.Module):
        def __init__(self, input_dim: int, encoding_dim: int = 5, hidden_dim: int = 64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, encoding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    def __init__(self, 
           tickers: list,
           start: str = "2019-01-01", 
           end: str = "2025-01-01",
           full_data_file: str = "data/russel_data_full.csv", # this will make this much faster
           train_file: str = "data/russel_data_train.csv",
           test_file: str = "data/russel_data_test.csv",
           autoencoder_params: dict = None, 
           clustering_params: dict = None,
           cointegration_threshold: float = 0.05, 
           hurst_threshold: float = 0.5,
           min_cluster_size: int = 3, 
           split_ratio: float = 0.5,
           log_level: int = logging.INFO):
      """
      Initialize PairsFinder with customization options.
      """
      self.tickers = tickers
      self.start = start
      self.end = end
      self.full_data_file = full_data_file
      self.train_file = train_file
      self.test_file = test_file
      self.split_ratio = split_ratio

      self.cointegration_threshold = cointegration_threshold
      self.hurst_threshold = hurst_threshold
      self.min_cluster_size = min_cluster_size

      self.autoencoder_params = autoencoder_params if autoencoder_params is not None else {
        'encoding_dim': 5, 'num_epochs': 1000, 'lr': 0.01, 'print_every': 50, 'hidden_dim': 64
      }
      self.clustering_params = clustering_params if clustering_params is not None else {
        'num_clusters': None, 'random_state': 42
      }

      self.full_data = None
      self.train_data = None
      self.test_data = None
      self.scaled_returns = None
      self.asset_names = None
      self.encoded_returns = None
      self.asset_clusters = None
      self.pairs_df = pd.DataFrame()

      logging.basicConfig(level=log_level, force=True)
      self.logger = logging.getLogger(__name__)

    def load_full_data(self) -> pd.DataFrame:
        """Load full data from CSV if it exists; otherwise, download and save."""
        if os.path.exists(self.full_data_file):
            self.full_data = pd.read_csv(self.full_data_file, parse_dates=['Date'], index_col='Date')
            self.logger.info("Loaded full data from file.")
        else:
            self.logger.info("Downloading full data ...")
            data = yf.download(self.tickers, start=self.start, end=self.end)['Close']
            data.to_csv(self.full_data_file)
            self.full_data = data
            self.logger.info("Downloaded and saved full data.")
        return self.full_data

    def split_data(self) -> tuple:
        """Split full data into train and test sets and save them."""
        if self.full_data is None:
            self.load_full_data()
        split_idx = int(self.full_data.shape[0] * self.split_ratio)
        self.train_data = self.full_data.iloc[:split_idx].copy()
        self.test_data = self.full_data.iloc[split_idx:].copy()
        self.train_data.to_csv(self.train_file)
        self.test_data.to_csv(self.test_file)
        self.logger.info(f"Data split into train ({split_idx} rows) and test, and saved.")
        return self.train_data, self.test_data

    def preprocess_returns(self) -> np.ndarray:
        """
        Calculate percent returns, standardize, and filter out assets with all-zero returns.
        Returns:
            Scaled returns as a NumPy array with assets as rows.
        """
        returns = self.train_data.pct_change().iloc[1:].fillna(0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(returns)
        # Transpose so that each asset is a row
        scaled = scaled.T
        valid_idx = np.where(~np.all(scaled == 0, axis=1))[0]
        self.scaled_returns = scaled[valid_idx]
        self.asset_names = returns.columns[valid_idx]
        self.logger.info(f"Preprocessed returns for {self.scaled_returns.shape[0]} assets.")
        return self.scaled_returns

    def train_autoencoder(self) -> tuple:
        """
        Train an autoencoder on the scaled returns and obtain the encoded representation.
        Returns:
            Tuple of encoded returns (as a NumPy array) and the trained model.
        """
        if self.scaled_returns is None:
            self.preprocess_returns()
        X_train = torch.tensor(self.scaled_returns, dtype=torch.float32)
        input_dim = X_train.shape[1]
        encoding_dim = self.autoencoder_params.get('encoding_dim', 5)
        hidden_dim = self.autoencoder_params.get('hidden_dim', 64)
        model = self.Autoencoder(input_dim, encoding_dim, hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=self.autoencoder_params.get('lr', 0.01))
        criterion = nn.MSELoss()
        num_epochs = self.autoencoder_params.get('num_epochs', 1000)
        print_every = self.autoencoder_params.get('print_every', 50)

        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            reconstructed = model(X_train)
            loss = criterion(reconstructed, X_train)
            loss.backward()
            optimizer.step()
            if epoch % print_every == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        model.eval()
        with torch.no_grad():
            self.encoded_returns = model.encoder(X_train).numpy()
        self.logger.info("Autoencoder training complete.")
        return self.encoded_returns, model

    def cluster_assets(self) -> pd.DataFrame:
        """
        Cluster assets using KMeans on the encoded returns.
        Returns:
            DataFrame mapping each asset to its cluster.
        """
        if self.encoded_returns is None:
            self.train_autoencoder()
        num_clusters = self.clustering_params.get('num_clusters', None)
        if num_clusters is None:
            num_clusters = max(1, int(self.encoded_returns.shape[0] / 4))
        kmeans = KMeans(n_clusters=num_clusters, random_state=self.clustering_params.get('random_state', 42))
        clusters = kmeans.fit_predict(self.encoded_returns)
        self.asset_clusters = pd.DataFrame({
            "Asset": self.asset_names,
            "Cluster": clusters
        })
        self.logger.info(f"Assets clustered into {num_clusters} clusters.")
        return self.asset_clusters

    @staticmethod
    def calculate_hurst_exponent(series: np.ndarray) -> float:
        """
        Calculate the Hurst exponent for a given series.
        Args:
            series: 1D NumPy array representing a time series.
        Returns:
            Hurst exponent or None if calculation fails.
        """
        try:
            H, _, _ = compute_Hc(series, kind='price', simplified=True)
            return H
        except Exception:
            return None

    def find_cointegrated_pairs(self) -> pd.DataFrame:
        """
        Find cointegrated and mean-reverting pairs from clustered assets.
        Processes clusters in order by cluster ID.
        Returns:
            DataFrame containing cointegrated pairs and quality scores.
        """
        if self.asset_clusters is None:
            self.cluster_assets()
        cointegrated_pairs = []
        for cluster_id in sorted(self.asset_clusters["Cluster"].unique()):
            cluster_assets = self.asset_clusters[self.asset_clusters["Cluster"] == cluster_id]
            if len(cluster_assets) < self.min_cluster_size:
                continue
            assets = cluster_assets["Asset"].tolist()
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    asset1, asset2 = assets[i], assets[j]
                    series1 = self.train_data[asset1].replace([np.inf, -np.inf], np.nan).ffill()
                    series2 = self.train_data[asset2].replace([np.inf, -np.inf], np.nan).ffill()
                    if series1.isnull().any() or series2.isnull().any():
                        self.logger.warning(f"Skipping pair {asset1}-{asset2} due to missing values")
                        continue
                    score, p_value, _ = coint(series1, series2)
                    if p_value < self.cointegration_threshold:
                        spread = series1 - series2
                        H = self.calculate_hurst_exponent(spread.values)
                        if H is not None and H < self.hurst_threshold:
                            cointegrated_pairs.append({
                                'Asset1': asset1,
                                'Asset2': asset2,
                                'P-Value': p_value,
                                'Hurst': H,
                                'Spread_Std': np.std(spread)
                            })
            self.logger.info(f"Cluster {cluster_id} ({len(cluster_assets)} assets): Found {len(cointegrated_pairs)} valid pairs so far.")
        if cointegrated_pairs:
            self.pairs_df = pd.DataFrame(cointegrated_pairs)
            ## adding a clip (0.05, 0.95) to avoid zero weight for certain pairs or too high weight
            self.pairs_df['P-Value_adj'] = (
                (self.pairs_df['P-Value'] - self.pairs_df['P-Value'].min()) /
                (self.pairs_df['P-Value'].max() - self.pairs_df['P-Value'].min() + 1e-8)
            )
            self.pairs_df['Hurst_adj'] = (
                (self.pairs_df['Hurst'] - self.pairs_df['Hurst'].min() + 0.05) /
                (self.pairs_df['Hurst'].max() - self.pairs_df['Hurst'].min() + 1e-8)
            )
            self.pairs_df['Quality_Score'] = (1 - self.pairs_df['P-Value_adj']) * (1 - self.pairs_df['Hurst_adj']).clip(lower=0.05, upper=0.95)
            self.pairs_df.sort_values('Quality_Score', ascending=False, inplace=True)
            self.pairs_df.reset_index(drop=True, inplace=True)
            self.logger.info("Cointegrated pairs found and quality scored.")
        else:
            self.logger.info("No valid cointegrated pairs found meeting the criteria.")
        return self.pairs_df

    def run_pipeline(self) -> pd.DataFrame:
        """Run the full pipeline: load, split, preprocess, encode, cluster, and find pairs."""
        self.logger.info("Starting pairs finding pipeline...")
        self.load_full_data()
        self.logger.info("Full data loaded.")
        self.split_data()
        self.logger.info("Data split into train and test sets.")
        self.preprocess_returns()
        self.logger.info("Returns preprocessed.")
        self.train_autoencoder()
        self.logger.info("Autoencoder trained.")
        self.cluster_assets()
        self.logger.info("Assets clustered.")
        self.find_cointegrated_pairs()
        self.logger.info("Cointegrated pairs found.")
        return self.pairs_df

    def save_pairs(self, filename: str = "data/pairs_df.csv", index: bool = False) -> None:
        """Save the found pairs to a CSV file."""
        if not self.pairs_df.empty:
            self.pairs_df.to_csv(filename, index=index)
            self.logger.info(f"Pairs saved to {filename}.")
        else:
            self.logger.warning("No pairs to save.")
            
        return None


if __name__ == "__main__":
  
    tickers = pd.read_csv('data/russel3000_stocks.csv')['Ticker'].tolist()
  
    # # easy usage:
    # pairs_finder = PairsFinder(tickers)
    # pairs_finder.run_pipeline()
    # pairs_finder.save_pairs("data/cointegrated_pairs.csv")
    
    
    ## more custom usage: (longer runtime)
    autoencoder_params = {
      'encoding_dim': 10, # 10D latent space
      'num_epochs': 500,
      'lr': 0.005, # learnign rate
      'print_every': 25, 
      'hidden_dim': 128 
    }
    
    clustering_params = {
    'num_clusters': 100, # more clusters will make code run faster
    'random_state': 1
    }

    cointegration_threshold = 0.05
    hurst_threshold = 0.45
    min_cluster_size = 4
    split_ratio = 0.6
    
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
        log_level=logging.DEBUG  # use logging.DEBUG for more detail / loggin.INFO for less
    )
    pf.run_pipeline()
    pf.save_pairs("data/100_clusters_pairs.csv")
        
    
    
    
