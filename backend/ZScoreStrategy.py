import numpy as np
import pandas as pd
import warnings
import argparse

from backend.BaseBacktest import BaseBacktest

warnings.filterwarnings("ignore")

class ZScoreStrategy(BaseBacktest):
    """
    A pairs trading strategy based on the z-score of the spread between two assets.
    
    This strategy calculates the hedge ratio using OLS regression on log-transformed prices,
    computes the spread and its rolling mean and standard deviation to derive a z-score, and then
    generates trading signals based on fixed thresholds applied to the z-score. Positions are entered 
    when the z-score exceeds an upper or lower threshold and exited when the z-score reverts or when 
    stop-loss/take-profit conditions are met. Strategy returns for each pair are aggregated, optionally
    using quality-based weighting.
    
    Command Line Args:
        --no-plot: Disable plotting of portfolio performance.
        --pairs-file: Path to the pairs file (default: data/cointegrated_pairs.csv).
        --help: Show this help message and exit.
    """
    
    def __init__(self, pairs_df, test_data, full_data=None, capital=1.0,
                 upper_threshold=1.28, lower_threshold=-1.28, exit_threshold=0.4,
                 stop_loss_pct=0.05, take_profit_pct=0.2):
        super().__init__(pairs_df, test_data, full_data, capital)
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.portfolio_test_returns = []
    
    def run(self):
        for idx, row in self.pairs_df.iterrows():
            ticker1, ticker2 = row['Asset1'], row['Asset2']
            quality_weight = row["Scaled_Quality"]
            pair_df = self.prepare_pair_data(ticker1, ticker2, self.test_data, min_length=30, log_transform=True)
            
            if pair_df is None or len(pair_df) < 30:
                continue
            try:
                hedge_ratio = self.calculate_hedge_ratio(pair_df, ticker1, ticker2, use_log=True)
                pair_df = self.compute_spread_and_zscore(pair_df, ticker1, ticker2, hedge_ratio)
                pair_df['Position'] = 0
                pair_df['Entry_Value'] = np.nan
                pair_df['Current_Value'] = np.nan
                position_state = 0
                entry_value = 0

                for i in range(1, len(pair_df)):
                    current_z = pair_df['z_score'].iloc[i]
                    prev_pos = pair_df['Position'].iloc[i - 1]
                    ret1 = pair_df[ticker1].iloc[i] / pair_df[ticker1].iloc[i - 1] - 1
                    ret2 = pair_df[ticker2].iloc[i] / pair_df[ticker2].iloc[i - 1] - 1
                    hedged_ret = ret2 - hedge_ratio * ret1

                    if prev_pos == 0:
                        if current_z > self.upper_threshold:
                            position_state = -1  # Short spread
                            entry_value = 1.0
                        elif current_z < self.lower_threshold:
                            position_state = 1   # Long spread
                            entry_value = 1.0
                        if position_state != 0:
                            pair_df.loc[pair_df.index[i], 'Entry_Value'] = entry_value
                    else:
                        current_value = entry_value * (1 + hedged_ret * prev_pos)
                        pair_df.loc[pair_df.index[i], 'Current_Value'] = current_value
                        if current_value < 1 - self.stop_loss_pct:
                            position_state = 0
                            pair_df.loc[pair_df.index[i], 'Current_Value'] = 1 - self.stop_loss_pct
                        elif current_value > 1 + self.take_profit_pct:
                            position_state = 0
                            pair_df.loc[pair_df.index[i], 'Current_Value'] = 1 + self.take_profit_pct
                        elif (prev_pos == -1 and current_z < self.exit_threshold) or \
                             (prev_pos == 1 and current_z > -self.exit_threshold):
                            position_state = 0
                    
                    pair_df.loc[pair_df.index[i], 'Position'] = position_state
                    
                    if position_state == 0:
                        entry_value = 0

                pair_df['hedged_returns'] = pair_df[ticker2].pct_change() - hedge_ratio * pair_df[ticker1].pct_change()
                pair_df['strategy_return'] = pair_df['hedged_returns'] * pair_df['Position'].shift(1)
                pair_df['capital'] = self.capital * (1 + pair_df['strategy_return']).cumprod()

                self.pair_results[f'{ticker1}_{ticker2}'] = pair_df
                self.portfolio_test_returns.append(pair_df['strategy_return'] * quality_weight)
            
            except Exception as e:
                print(f"Error processing {ticker1}-{ticker2}: {e}")
                continue
        
        if self.portfolio_test_returns:
            portfolio_returns = pd.concat(self.portfolio_test_returns, axis=1).sum(axis=1)
            self.portfolio_capital = (1 + portfolio_returns).cumprod()
            self.portfolio_returns = portfolio_returns

            print("\nZ-Score Strategy Performance Metrics:")
            print(f"Final Return: {self.portfolio_capital.iloc[-1] - 1:.2%}")
            sharpe = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                      if portfolio_returns.std() != 0 else 0)
            drawdown = (self.portfolio_capital / self.portfolio_capital.cummax() - 1).min()
            print(f"Annualized Sharpe: {sharpe:.2f}")
            print(f"Max Drawdown: {drawdown:.2%}")
            self.plot_portfolio_performance(title="Pairs Trading Performance (Z-Score Strategy)")
        else:
            print("No valid pairs processed for the Z-Score strategy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Z-Score Strategy Backtest with optional plotting.")
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

    print("Running Z-Score Strategy Backtest...")
    zscore_strategy = ZScoreStrategy(pairs_df, test_data, full_data=full_data, capital=100)
    
    if args.no_plot:
        zscore_strategy.plot_portfolio_performance = lambda title="": None

    zscore_strategy.run()
    
    zscore_strategy.save_results("docs/strategy-examples/zscore_results.csv")
