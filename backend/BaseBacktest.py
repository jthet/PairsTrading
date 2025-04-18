import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import argparse

class BaseBacktest:
    def __init__(self, pairs_df, test_data, full_data=None, capital=100):
        """
        Base backtest class for pairs trading.
        
        Parameters:
            pairs_df (pd.DataFrame): DataFrame with at least columns 'Asset1' and 'Asset2'. 
                                     Optionally, include a 'Quality_Score' column for weighting.
            test_data (pd.DataFrame): Price data (e.g., historical prices) used for simpler backtests.
            full_data (pd.DataFrame): Price data used for model (ML) strategies. If not provided, full_data=test_data.
            capital (float): Starting capital per pair.
            
        Command Line Args:
          --no-plot: Disable plotting of portfolio performance.
          --pairs-file: Path to the pairs file (default: data/cointegrated_pairs.csv).
          --help: Show this help message and exit.
        """
        self.pairs_df = pairs_df.copy()
        self.test_data = test_data.copy()
        self.full_data = full_data.copy() if full_data is not None else test_data.copy()
        self.capital = capital
        self.pair_results = {}
        self.portfolio_returns = None
        self.portfolio_capital = None
        
        # Use quality scores for weighting if provided... 
        if "Quality_Score" in self.pairs_df.columns:
            self.pairs_df["Scaled_Quality"] = self.pairs_df["Quality_Score"].clip(lower=0.05, upper=0.95) / self.pairs_df["Quality_Score"].clip(lower=0.05, upper=0.95).sum()
        else:
            self.pairs_df["Scaled_Quality"] = 1.0 / len(self.pairs_df)
    
    def prepare_pair_data(self, ticker1, ticker2, data, min_length=30, log_transform=True):
        """
        Extract and clean data for a given pair.
        """
        pair_df = data.loc[:, [ticker1, ticker2]].copy()
        pair_df = pair_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        if len(pair_df.dropna()) < min_length:
            return None
          
        if log_transform:
            with np.errstate(all='ignore'):
                pair_df[f'log_{ticker1}'] = np.log(pair_df[ticker1])
                pair_df[f'log_{ticker2}'] = np.log(pair_df[ticker2])
            pair_df = pair_df.dropna()
        
        return pair_df
    
    def calculate_hedge_ratio(self, pair_df, ticker1, ticker2, use_log=True):
        """
        Calculate hedge ratio using OLS regression.
        """
        if use_log:
            y = pair_df[f'log_{ticker2}']
            x = pair_df[f'log_{ticker1}']
        else:
            y = pair_df[ticker2]
            x = pair_df[ticker1]
        model = sm.OLS(y, sm.add_constant(x)).fit()
        return model.params.iloc[1]
    
    def compute_spread_and_zscore(self, pair_df, ticker1, ticker2, hedge_ratio, lookback=None):
        """
        Compute spread, rolling mean and standard deviation, and derive the z-score.
        """
        # Use log prices if available
        if f'log_{ticker1}' in pair_df.columns and f'log_{ticker2}' in pair_df.columns:
            y = pair_df[f'log_{ticker2}']
            x = pair_df[f'log_{ticker1}']
        else:
            y = pair_df[ticker2]
            x = pair_df[ticker1]
        pair_df['spread'] = y - hedge_ratio * x
        
        if lookback is None:
            lookback = min(63, len(pair_df) // 2)
        
        pair_df['spread_mean'] = pair_df['spread'].rolling(lookback).mean()
        pair_df['spread_std'] = pair_df['spread'].rolling(lookback).std()
        pair_df['z_score'] = (pair_df['spread'] - pair_df['spread_mean']) / pair_df['spread_std']
        
        return pair_df
    
    def plot_portfolio_performance(self, title="Portfolio Performance"):
        """
        Plot the cumulative capital curve of the portfolio.
        """
        if self.portfolio_capital is not None:
            plt.figure(figsize=(14, 7))
            plt.plot(self.portfolio_capital, label="Portfolio Capital")
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Capital Growth")
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("No portfolio performance to plot.")
    
    def run(self):
        """
        Place holder for child methods.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def save_results(self, filename="results.csv"):
        """
        Save one pair's result as an example. You can extend this to save all pairs.
        """
        if self.pair_results:
            first_pair_key = list(self.pair_results.keys())[0]
            self.pair_results[first_pair_key].to_csv(filename)
            print(f"Results for pair {first_pair_key} saved to {filename}.")
        else:
            print("No results to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BaseBacktest with optional plotting.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting of portfolio performance.")
    parser.add_argument("--pairs-file", type=str, default="data/cointegrated_pairs.csv",
                        help="Path to the pairs file (default: data/cointegrated_pairs.csv)")
    args = parser.parse_args()
    
    try:
        test_data = pd.read_csv("data/russel_data_test.csv", index_col=0, parse_dates=True)
        pairs_df = pd.read_csv(args.pairs_file)
        full_data = pd.read_csv("data/russel_data_full.csv", index_col=0, parse_dates=True)
    except Exception as e:
        print("Error loading data:", e)
        exit(-1)
    
    backtester = BaseBacktest(pairs_df, test_data, full_data=full_data, capital=100)
    
    ticker_1 = pairs_df['Asset1'][0]
    ticker_2 = pairs_df['Asset2'][0]
    
    pair_df = backtester.prepare_pair_data(ticker_1, ticker_2, test_data, min_length=30, log_transform=True)
    
    if pair_df is not None:
        print("Sample prepared pair data:")
        print(pair_df.head())
        
        hedge_ratio = backtester.calculate_hedge_ratio(pair_df, ticker_1, ticker_2, use_log=True)
        print("\nCalculated Hedge Ratio:", hedge_ratio)
    
        pair_df = backtester.compute_spread_and_zscore(pair_df, ticker_1, ticker_2, hedge_ratio)
        print("\nSpread and z-score data:")
        print(pair_df[['spread', 'spread_mean', 'spread_std', 'z_score']].tail())
    else:
        print("Insufficient data for the pair.")
    
    if args.no_plot:
        backtester.plot_portfolio_performance = lambda title="": None

    backtester.plot_portfolio_performance(title="Dummy Portfolio Performance")
