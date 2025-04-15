import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from backend.BaseBacktest import BaseBacktest

# Load data
pairs_df = pd.read_csv("data/cointegrated_pairs.csv")
test_data = pd.read_csv("data/russel_data_test.csv", index_col=0, parse_dates=True)

# Streamlit interface
st.set_page_config(layout="wide")
st.title("Pairs Trading Dashboard")

# Select a pair
pair_options = [f"{row['Asset1']} - {row['Asset2']}" for _, row in pairs_df.iterrows()]
selected_pair = st.selectbox("Select a cointegrated pair", pair_options)

# Parse selection
asset1, asset2 = selected_pair.split(" - ")

# Run pipeline for the pair
backtester = BaseBacktest(pairs_df, test_data, capital=100)
pair_df = backtester.prepare_pair_data(asset1, asset2, test_data)
hedge_ratio = backtester.calculate_hedge_ratio(pair_df, asset1, asset2)
pair_df = backtester.compute_spread_and_zscore(pair_df, asset1, asset2, hedge_ratio)

# Display stats
st.subheader(f"Hedge Ratio: {hedge_ratio:.4f}")
st.metric("Mean Spread", f"{pair_df['spread'].mean():.4f}")
st.metric("Std Dev of Spread", f"{pair_df['spread'].std():.4f}")

# Plot spread
st.subheader("Spread Over Time")
fig, ax = plt.subplots()
ax.plot(pair_df.index, pair_df['spread'], label="Spread")
ax.axhline(pair_df['spread'].mean(), color="red", linestyle="--", label="Mean")
ax.set_title(f"{asset1} - {asset2} Spread")
ax.legend()
st.pyplot(fig)

# Plot z-score
st.subheader("Z-Score Over Time")
fig2, ax2 = plt.subplots()
ax2.plot(pair_df.index, pair_df['z_score'], label="Z-Score", color="purple")
ax2.axhline(1, color="green", linestyle="--")
ax2.axhline(-1, color="green", linestyle="--")
ax2.axhline(0, color="black", linestyle=":")
ax2.set_title("Z-score of Spread")
ax2.legend()
st.pyplot(fig2)
