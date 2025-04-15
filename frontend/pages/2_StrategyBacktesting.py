import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.sidebar.header("Strategy Selection")
strategy_choice = st.sidebar.selectbox(
    "Select a Strategy",
    ["ZScoreStrategy", "LogisticStrategy", "XGBoostStrategy"]
)

if strategy_choice == "ZScoreStrategy":
    from backend.ZScoreStrategy import ZScoreStrategy as StrategyClass
elif strategy_choice == "LogisticStrategy":
    from backend.LogisticStrategy import LogisticStrategy as StrategyClass
elif strategy_choice == "XGBoostStrategy":
    from backend.XGBoostStrategy import XGBoostStrategy as StrategyClass
else:
    st.error("Invalid Strategy Selected")
    st.stop()

def run():
    """Strategy Testing Page for Pairs Trading using the selected strategy."""
    st.title(f"Strategy Testing: {strategy_choice}")

    # loads the data... need to let the user do this eventually
    pairs_csv = os.path.join("data", "cointegrated_pairs.csv")
    test_data_csv = os.path.join("data", "russel_data_test.csv")
    full_data_csv = os.path.join("data", "russel_data_full.csv")

    try:
        pairs_df = pd.read_csv(pairs_csv)
        test_data = pd.read_csv(test_data_csv, index_col=0, parse_dates=True)
        full_data = pd.read_csv(full_data_csv, index_col=0, parse_dates=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    st.sidebar.header("Common Parameter")
    capital = st.sidebar.number_input("Initial Capital", value=100)

    if strategy_choice == "ZScoreStrategy":
        st.sidebar.header("Z-Score Strategy Parameters")
        upper_threshold = st.sidebar.number_input("Upper Threshold", value=1.28)
        lower_threshold = st.sidebar.number_input("Lower Threshold", value=-1.28)
        exit_threshold = st.sidebar.number_input("Exit Threshold", value=0.4)
        stop_loss_pct = st.sidebar.number_input("Stop Loss (%)", value=0.05)
        take_profit_pct = st.sidebar.number_input("Take Profit (%)", value=0.2)
    else:
        st.sidebar.header(f"{strategy_choice} Parameters")
        min_zscore_threshold = st.sidebar.number_input("Min Z-Score Threshold", value=0.5)
        min_proba_threshold = st.sidebar.number_input("Min Probability Threshold", value=0.6)

    if st.button("Run Strategy Test"):
        st.write(f"Running {strategy_choice} Backtest...")
        
        if strategy_choice == "ZScoreStrategy":
            strategy_instance = StrategyClass(
                pairs_df, 
                test_data, 
                full_data=full_data, 
                capital=capital,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                exit_threshold=exit_threshold,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct
            )
        elif strategy_choice == "LogisticStrategy":
            strategy_instance = StrategyClass(
                pairs_df, 
                test_data, 
                full_data=full_data, 
                capital=capital,
                min_zscore_threshold=min_zscore_threshold,
                min_proba_threshold=min_proba_threshold
            )
        elif strategy_choice == "XGBoostStrategy":
            strategy_instance = StrategyClass(
                pairs_df, 
                test_data, 
                full_data=full_data, 
                capital=capital,
                min_zscore_threshold=min_zscore_threshold,
                min_proba_threshold=min_proba_threshold
            )
        else:
            st.error("Unexpected strategy selection.")
            st.stop()

        # run strat
        strategy_instance.run()

        if hasattr(strategy_instance, "portfolio_capital"):
            portfolio_capital = strategy_instance.portfolio_capital
            st.write("### Final Results")
            final_return = portfolio_capital.iloc[-1] - 1
            st.write(f"Final Return: {final_return:.2%}")

            # plotting
            fig, ax = plt.subplots()
            ax.plot(portfolio_capital.index, portfolio_capital, label="Portfolio Capital")
            ax.set_title("Portfolio Performance")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No results found. Ensure the strategy's run() method populates the expected attributes.")

if __name__ == "__main__":
    run()
