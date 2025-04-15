import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from backend.BaseBacktest import BaseBacktest

pairs_csv = os.path.join("data", "cointegrated_pairs.csv")
test_data_csv = os.path.join("data", "russel_data_test.csv")

pairs_df = pd.read_csv(pairs_csv)
test_data = pd.read_csv(test_data_csv, index_col=0, parse_dates=True)

st.set_page_config(page_title="Spread Analysis", layout="wide")
st.title("Spread Analysis for Cointegrated Pairs")

pair_options = [f"{row['Asset1']} - {row['Asset2']}" for _, row in pairs_df.iterrows()]
selected_pair = st.selectbox("Select a cointegrated pair", pair_options)

asset1, asset2 = selected_pair.split(" - ")

backtester = BaseBacktest(pairs_df, test_data, capital=100)
pair_df = backtester.prepare_pair_data(asset1, asset2, test_data)
hedge_ratio = backtester.calculate_hedge_ratio(pair_df, asset1, asset2)
pair_df = backtester.compute_spread_and_zscore(pair_df, asset1, asset2, hedge_ratio)

st.subheader(f"Hedge Ratio: {hedge_ratio:.4f}")
st.metric("Mean Spread", f"{pair_df['spread'].mean():.4f}")
st.metric("Std Dev of Spread", f"{pair_df['spread'].std():.4f}")

st.subheader("Spread Over Time")
fig, ax = plt.subplots()
ax.plot(pair_df.index, pair_df['spread'], label="Spread")
ax.axhline(pair_df['spread'].mean(), color="red", linestyle="--", label="Mean")
ax.set_title(f"{asset1} - {asset2} Spread")
ax.legend()
st.pyplot(fig)

st.subheader("Z-Score Over Time")
fig2, ax2 = plt.subplots()
ax2.plot(pair_df.index, pair_df['z_score'], label="Z-Score", color="purple")
ax2.axhline(1, color="green", linestyle="--")
ax2.axhline(-1, color="green", linestyle="--")
ax2.axhline(0, color="black", linestyle=":")
ax2.set_title("Z-score of Spread")
ax2.legend()
st.pyplot(fig2)
