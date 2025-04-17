import sys, os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from backend.BaseBacktest import BaseBacktest

st.set_page_config(
    page_title="Spread Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("🔍 Spread Analysis for Cointegrated Pairs")

pairs_df = (
    st.session_state.get("pairs_df")
    or pd.read_csv("data/cointegrated_pairs.csv")
)
test_data = (
    st.session_state.get("test_data")
    or pd.read_csv("data/russel_data_test.csv", index_col=0, parse_dates=True)
)
train_data = (
    st.session_state.get("train_data")
    or pd.read_csv("data/russel_data_train.csv", index_col=0, parse_dates=True)
)

pair_options = [f"{a} – {b}" for a, b in zip(pairs_df.Asset1, pairs_df.Asset2)]
asset1, asset2 = st.selectbox("Choose a pair", pair_options).split(" – ")

bt = BaseBacktest(pairs_df, test_data, full_data=train_data)
df_pair = bt.prepare_pair_data(asset1, asset2, test_data)
hr = bt.calculate_hedge_ratio(df_pair, asset1, asset2)
df_pair = bt.compute_spread_and_zscore(df_pair, asset1, asset2, hr)

# metrics
st.subheader(f"Hedge Ratio: {hr:.4f}")
m1, m2 = st.columns(2)
m1.metric("Mean Spread", f"{df_pair['spread'].mean():.4f}")
m2.metric("Std Dev of Spread", f"{df_pair['spread'].std():.4f}")

# spread chart
spread_fig = px.line(
    df_pair,
    x=df_pair.index,
    y="spread",
    title=f"{asset1} – {asset2} Spread",
    labels={"spread": "Spread", "index": "Date"},
)
spread_fig.add_hline(
    y=df_pair['spread'].mean(),
    line_dash="dash",
    line_color="red",
    annotation_text="Mean",
    annotation_position="bottom right",
)

# zscore chart
z_fig = px.line(
    df_pair,
    x=df_pair.index,
    y="z_score",
    title="Z‑Score of Spread",
    labels={"z_score": "Z‑Score", "index": "Date"},
)
# thresholds
for lvl in [1, -1]:
    z_fig.add_hline(
        y=lvl,
        line_dash="dash",
        line_color="green",
        annotation_text=f"{lvl:+}",
        annotation_position="top right" if lvl>0 else "bottom right",
    )
z_fig.add_hline(y=0, line_dash="dot", line_color="black")


# side by side
c1, c2 = st.columns(2, gap="large")
c1.plotly_chart(spread_fig, use_container_width=True)
c2.plotly_chart(z_fig, use_container_width=True)
