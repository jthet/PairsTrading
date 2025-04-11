import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from BaseBacktest import BaseBacktest
import argparse

class LogisticStrategy(BaseBacktest):
    """
    A pairs trading strategy that uses a Logistic Regression model to generate trading signals.
    
    This model uses engineered features from the asset pair's spread (like z_score, moving averages,
    momentum, and volatility) to predict the likelihood of a favorable future spread move.
    The logistic regression's predicted probability is then used (with a threshold) to trigger trade entries and exits.
    
    Command Line Args:
        --no-plot: Disable plotting of portfolio performance.
        --pairs-file: Path to the pairs file (default: data/cointegrated_pairs.csv).
        --help: Show this help message and exit.
    """
  
    def __init__(self, pairs_df, test_data, full_data, capital=1.0, logreg_params=None,
                 min_zscore_threshold=0.5, min_proba_threshold=0.6):
        """
        Logistic Regression–based strategy.
        
        Parameters:
            logreg_params (dict): Parameters for the logistic regression model.
            min_zscore_threshold (float): Minimum absolute z-score to trigger trades.
            min_proba_threshold (float): Minimum probability threshold used for entry conditions.
        """
        super().__init__(pairs_df, test_data, full_data, capital)
        self.logreg_params = logreg_params if logreg_params is not None else {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42
        }
        self.min_zscore_threshold = min_zscore_threshold
        self.min_proba_threshold = min_proba_threshold
        self.portfolio_test_returns = []
    
    def run(self):
        """
        Execute the Logistic Regression-based pairs trading strategy.
        
        For each asset pair, the function computes the hedge ratio, spread, and z_score,
        then trains a Logistic Regression model using engineered features. The model produces
        predicted probabilities that are compared against thresholds to generate trading signals.
        The strategy then computes returns and portfolio performance metrics.
        """
        # Use full_data for model training and signal generation.
        for idx, row in self.pairs_df.iterrows():
            ticker1, ticker2 = row['Asset1'], row['Asset2']
            quality_weight = row["Scaled_Quality"]
            pair_df = self.prepare_pair_data(ticker1, ticker2, self.full_data, min_length=200, log_transform=True)
            if pair_df is None or len(pair_df) < 200:
                print(f"Skipping {ticker1}-{ticker2}: insufficient data")
                continue
            try:
                hedge_ratio = self.calculate_hedge_ratio(pair_df, ticker1, ticker2, use_log=True)
                pair_df = self.compute_spread_and_zscore(pair_df, ticker1, ticker2, hedge_ratio)
                pair_df['ma_10'] = pair_df['spread'].rolling(10).mean()
                pair_df['ma_30'] = pair_df['spread'].rolling(30).mean()
                pair_df['momentum'] = pair_df['spread'] - pair_df['spread'].shift(5)
                pair_df['volatility'] = pair_df['spread'].rolling(20).std()
                pair_df['hedged_returns'] = pair_df[ticker2].pct_change() - hedge_ratio * pair_df[ticker1].pct_change()
                pair_df['target'] = (pair_df['z_score'].shift(-20) - pair_df['z_score'] > 0).astype(int)
                pair_df = pair_df.dropna()

                features = ['z_score', 'ma_10', 'ma_30', 'momentum', 'volatility', 'spread_mean', 'spread_std']
                X = pair_df[features]
                y = pair_df['target']

                split_idx = int(len(pair_df) * 0.5)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                lr_model = LogisticRegression(**self.logreg_params)
                lr_model.fit(X_train_scaled, y_train)
                train_preds = lr_model.predict(X_train_scaled)
                test_preds = lr_model.predict(X_test_scaled)
                train_acc = accuracy_score(y_train, train_preds)
                test_acc = accuracy_score(y_test, test_preds)
                print(f"\n{ticker1}-{ticker2} Logistic Regression Model Performance:")
                print(f"Train Accuracy: {train_acc:.2%}")
                print(f"Test Accuracy: {test_acc:.2%}")

                test_df = pair_df.iloc[split_idx:].copy()
                X_test_all = test_df[features]
                X_test_scaled_all = scaler.transform(X_test_all)
                test_df['pred_proba'] = lr_model.predict_proba(X_test_scaled_all)[:, 1]
                test_df['prediction'] = lr_model.predict(X_test_scaled_all)

                test_df['Position'] = 0
                position_state = 0
                for i in range(1, len(test_df)):
                    current_pred = test_df['prediction'].iloc[i]
                    current_proba = test_df['pred_proba'].iloc[i]
                    current_z = test_df['z_score'].iloc[i]
                    short_prob = 1 - current_proba  # probability for class 0
                    long_prob = current_proba       # probability for class 1
                    if position_state == 0:
                        if (short_prob > self.min_proba_threshold and
                            current_z > self.min_zscore_threshold and
                            current_pred == 0):
                            position_state = -1
                        elif (long_prob > self.min_proba_threshold and
                              current_z < -self.min_zscore_threshold and
                              current_pred == 1):
                            position_state = 1
                    else:
                        if ((position_state == -1 and (current_z < 0.1 or short_prob < self.min_proba_threshold)) or
                            (position_state == 1 and (current_z > -0.1 or long_prob < self.min_proba_threshold))):
                            position_state = 0
                    test_df.loc[test_df.index[i], 'Position'] = position_state

                test_df['strategy_return'] = test_df['hedged_returns'] * test_df['Position'].shift(1)
                test_df['capital'] = self.capital * (1 + test_df['strategy_return']).cumprod()
                self.pair_results[f'{ticker1}_{ticker2}'] = test_df
                self.portfolio_test_returns.append(test_df['strategy_return'] * quality_weight)
            except Exception as e:
                print(f"Error processing {ticker1}-{ticker2}: {e}")
                continue

        if self.portfolio_test_returns:
            portfolio_returns = pd.concat(self.portfolio_test_returns, axis=1).sum(axis=1)
            self.portfolio_capital = (1 + portfolio_returns).cumprod()
            self.portfolio_returns = portfolio_returns
            print("\nLogistic Regression Strategy Performance Metrics:")
            print(f"Final Return: {self.portfolio_capital.iloc[-1] - 1:.2%}")
            sharpe = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                      if portfolio_returns.std() != 0 else 0)
            drawdown = (self.portfolio_capital / self.portfolio_capital.cummax() - 1).min()
            print(f"Annualized Sharpe: {sharpe:.2f}")
            print(f"Max Drawdown: {drawdown:.2%}")
            self.plot_portfolio_performance(title="Pairs Trading Performance (Logistic Regression Strategy)")
        else:
            print("No valid pairs processed for the Logistic Regression strategy.")
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Logistic Regression Strategy Backtest with optional plotting.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting of portfolio performance.")
    parser.add_argument("--pairs-file", type=str, default="data/cointegrated_pairs.csv",
                        help="Path to the pairs file (default: data/cointegrated_pairs.csv)")
    args = parser.parse_args()
    
    try:
        pairs_df = pd.read_csv(args.pairs_file)
        test_data = pd.read_csv("data/russel_data_test.csv", index_col=0, parse_dates=True)
        full_data = pd.read_csv("data/russel_data_full.csv", index_col=0, parse_dates=True)
    except Exception as e:
        print("Error loading data:", e)
        exit(-1)
    
    print("Running Logistic Regression Strategy Backtest...")
    logistic_strategy = LogisticStrategy(pairs_df, test_data, full_data=full_data, capital=100)
    
    if args.no_plot:
        logistic_strategy.plot_portfolio_performance = lambda title="": None
        
    logistic_strategy.run()
    
    logistic_strategy.save_results("docs/strategy-examples/logreg_results.csv")
