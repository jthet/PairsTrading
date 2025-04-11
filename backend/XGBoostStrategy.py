import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
import argparse
import sys
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

from BaseBacktest import BaseBacktest

class XGBoostStrategy(BaseBacktest):
    """
    A pairs trading strategy that uses an XGBoost classifier to generate trading signals.
    
    This model is trained on engineered features (e.g., z_score, moving averages, momentum, volatility)
    extracted from the spread between two assets. It predicts the probability of a favorable spread move
    over a future time window. Trading entries and exits are then determined based on the predicted
    probabilities and additional z_score-based thresholds.
    
    Command Line Args:
        --no-plot: Disable plotting of portfolio performance.
        --help: Show this help message and exit.
    """
    
    def __init__(self, pairs_df, test_data, full_data, capital=1.0, xgb_params=None,
                 min_zscore_threshold=0.5, min_proba_threshold=0.6):
        """
        XGBoost-based strategy.
        
        Parameters:
            xgb_params (dict): Dictionary of parameters for the XGBoost model.
            min_zscore_threshold (float): Minimum absolute z-score to consider a trade.
            min_proba_threshold (float): Minimum probability threshold to trigger a position.
        """
        super().__init__(pairs_df, test_data, full_data, capital)
        self.xgb_params = xgb_params if xgb_params is not None else {
            'objective': 'binary:logistic',
            'max_depth': 2,
            'learning_rate': 0.01,
            'n_estimators': 400,
            'early_stopping_rounds': 10,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        self.min_zscore_threshold = min_zscore_threshold
        self.min_proba_threshold = min_proba_threshold
        self.portfolio_test_returns = []
    
    def run(self):
        
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
                pair_df['asset1_returns'] = pair_df[ticker1].pct_change()
                pair_df['asset2_returns'] = pair_df[ticker2].pct_change()
                pair_df['hedged_returns'] = pair_df['asset2_returns'] - hedge_ratio * pair_df['asset1_returns']
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

                xgb_model = xgb.XGBClassifier(**self.xgb_params)
                xgb_model.fit(X_train_scaled, y_train,
                              eval_set=[(X_test_scaled, y_test)],
                              verbose=False)

                train_preds = xgb_model.predict(X_train_scaled)
                test_preds = xgb_model.predict(X_test_scaled)
                train_acc = accuracy_score(y_train, train_preds)
                test_acc = accuracy_score(y_test, test_preds)
                print(f"\n{ticker1}-{ticker2} XGBoost Model Performance:")
                print(f"Train Accuracy: {train_acc:.2%}")
                print(f"Test Accuracy: {test_acc:.2%}")

                test_df = pair_df.iloc[split_idx:].copy()
                X_test_all = test_df[features]
                X_test_scaled_full = scaler.transform(X_test_all)
                test_df['pred_proba'] = xgb_model.predict_proba(X_test_scaled_full)[:, 1]
                test_df['prediction'] = xgb_model.predict(X_test_scaled_full)

                test_df['Position'] = 0
                position_state = 0
                for i in range(1, len(test_df)):
                    current_pred = test_df['prediction'].iloc[i]
                    current_proba = test_df['pred_proba'].iloc[i]
                    current_z = test_df['z_score'].iloc[i]
                    if position_state == 0:
                        if (current_proba > self.min_proba_threshold and
                            current_z > self.min_zscore_threshold and
                            current_pred == 0):
                            position_state = -1
                        elif (current_proba > 1 - self.min_proba_threshold and
                              current_z < -self.min_zscore_threshold and
                              current_pred == 1):
                            position_state = 1
                    elif position_state != 0:
                        if ((position_state == -1 and current_z < 0.3) or
                            (position_state == 1 and current_z > -0.3)):
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

            print("\nXGBoost Strategy Performance Metrics:")
            print(f"Final Return: {self.portfolio_capital.iloc[-1] - 1:.2%}")
            sharpe = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                      if portfolio_returns.std() != 0 else 0)
            drawdown = (self.portfolio_capital / self.portfolio_capital.cummax() - 1).min()
            print(f"Annualized Sharpe: {sharpe:.2f}")
            print(f"Max Drawdown: {drawdown:.2%}")
            self.plot_portfolio_performance(title="Pairs Trading Performance (XGBoost Strategy)")
        
        else:
            print("No valid pairs processed for the XGBoost strategy.")
            

if __name__ == "__main__":
    # Parse the command line args.
    parser = argparse.ArgumentParser(description="Run XGBoost Strategy Backtest with optional plotting.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting of portfolio performance.")
    args = parser.parse_args()
  
    try:
        pairs_df = pd.read_csv("data/cointegrated_pairs.csv")
        test_data = pd.read_csv("data/russel_data_test.csv", index_col=0, parse_dates=True)
        full_data = pd.read_csv("data/russel_data_full.csv", index_col=0, parse_dates=True)
    except Exception as e:
        print("Error loading data:", e)
        exit(-1)

    print("Running XGBoost Strategy Backtest...")
    xgboost_strat = XGBoostStrategy(pairs_df, test_data, full_data=full_data, capital=100)
    
    if args.no_plot:
        xgboost_strat.plot_portfolio_performance = lambda title="": None

    xgboost_strat.run()
    
    xgboost_strat.save_results("docs/strategy-examples/xgboost_results.csv")
