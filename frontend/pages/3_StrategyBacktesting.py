import sys, os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Pairs Trading Backtest",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.header("Select Strategy")

strategy_choice = st.sidebar.selectbox(
    "Strategy",
    ["ZScoreStrategy", "LogisticStrategy", "XGBoostStrategy"]
)

if strategy_choice == "ZScoreStrategy":
    from backend.ZScoreStrategy import ZScoreStrategy as StrategyClass
elif strategy_choice == "LogisticStrategy":
    from backend.LogisticStrategy import LogisticStrategy as StrategyClass
else:
    from backend.XGBoostStrategy import XGBoostStrategy as StrategyClass

# ─── load data (cached) ───────────────────────────────────────────────────────
@st.cache_data
def load_df(path, **kwargs):
    print(f"Loading {path}... NOT using cache")
    return pd.read_csv(path, **kwargs)

# pairs_df
if "pairs_df" in st.session_state:
    pairs_df = st.session_state["pairs_df"]
    print("pairs_df loaded from session state")
else:
    pairs_df = load_df("data/cointegrated_pairs.csv")

# full_data
if "full_data" in st.session_state:
    full_data = st.session_state["full_data"]
    print("full_data loaded from session state")
else:
    full_data = load_df(
        "data/russel_data_full.csv", index_col=0, parse_dates=True
    )

# test_data
if "test_data" in st.session_state:
    test_data = st.session_state["test_data"]
    print("test_data loaded from session state")
else:
    test_data = load_df(
        "data/russel_data_test.csv", index_col=0, parse_dates=True
    )

# train_data
if "train_data" in st.session_state:
    train_data = st.session_state["train_data"]
    print("train_data loaded from session state")
else:
    train_data = load_df(
        "data/russel_data_train.csv", index_col=0, parse_dates=True
    )


# sidebar

st.sidebar.markdown("---")
capital = st.sidebar.number_input("Initial Capital", value=100.0)

if strategy_choice == "ZScoreStrategy":
    with st.sidebar.expander("Z‑Score Params", expanded=False):
        upper = st.number_input("Upper Threshold", value=1.28)
        lower = st.number_input("Lower Threshold", value=-1.28)
        exit_t = st.number_input("Exit Threshold", value=0.4)
        sl = st.number_input("Stop‑loss (%)", value=0.05)
        tp = st.number_input("Take‑profit (%)", value=0.2)
else:
    with st.sidebar.expander(f"{strategy_choice} Params", expanded=False):
        min_z = st.number_input("Min Z‑Score", value=0.5)
        min_p = st.number_input("Min Probability", value=0.6)

st.sidebar.markdown("---")
run_backtest = st.sidebar.button("▶️ Run Backtest")

# strat select
st.title(f"🔄 Running: {strategy_choice}" if run_backtest else "Strategy Testing")

if run_backtest:
    with st.spinner(f"Backtesting {strategy_choice}…"):
        if strategy_choice == "ZScoreStrategy":
            strat = StrategyClass(
                pairs_df,
                test_data,
                full_data=train_data,
                capital=capital,
                upper_threshold=upper,
                lower_threshold=lower,
                exit_threshold=exit_t,
                stop_loss_pct=sl,
                take_profit_pct=tp
            )
        else:
            strat = StrategyClass(
                pairs_df,
                test_data,
                full_data=train_data,
                capital=capital,
                min_zscore_threshold=min_z,
                min_proba_threshold=min_p
            )

        strat.run()

    st.success("✅ Backtest complete!")

    #  results
    if hasattr(strat, "portfolio_capital"):
        port = strat.portfolio_capital
        final_ret = port.iloc[-1] / port.iloc[0] - 1
        c1, c2 = st.columns(2)
        c1.metric("Start Value", f"${port.iloc[0]*100:.2f}")
        c2.metric("Final Return", f"{final_ret:.2%}")

        # performance chart
        perf_fig = px.line(
            port,
            x=port.index,
            y=port.values * capital,
            labels={"y": "Portfolio Value", "x": "Date"},
            title="Portfolio Performance Over Time"
        )
        perf_fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(perf_fig, use_container_width=True)
    else:
        st.warning("No results found—check that your strategy’s `run()` populates `portfolio_capital`.")
