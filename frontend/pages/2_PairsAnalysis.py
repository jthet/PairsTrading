import sys, os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from backend.BaseBacktest import BaseBacktest

st.set_page_config(page_title="Spread Analysis", layout="wide", initial_sidebar_state="collapsed")
st.title("🔍 Spread Analysis for Cointegrated Pairs")

@st.cache_data
def load_df(path, **kwargs):
    return pd.read_csv(path, **kwargs)

if "pairs_df" in st.session_state:
    pairs_df = st.session_state["pairs_df"]
else:
    pairs_df = load_df("data/cointegrated_pairs.csv")


if "test_data" in st.session_state:
    test_data = st.session_state["test_data"]
else:
    test_data = load_df("data/russel_data_test.csv", index_col=0, parse_dates=True)

if "train_data" in st.session_state:
    train_data = st.session_state["train_data"]
else:
    train_data = load_df("data/russel_data_train.csv", index_col=0, parse_dates=True)

pair_options = [f"{a} – {b}" for a, b in zip(pairs_df.Asset1, pairs_df.Asset2)]
asset1, asset2 = st.selectbox("Choose a pair", pair_options).split(" – ")

bt = BaseBacktest(pairs_df, test_data, full_data=train_data)
df_pair = bt.prepare_pair_data(asset1, asset2, test_data)
hr = bt.calculate_hedge_ratio(df_pair, asset1, asset2)
df_pair = bt.compute_spread_and_zscore(df_pair, asset1, asset2, hr)

st.subheader(f"Hedge Ratio: {hr:.4f}")
c1, c2 = st.columns(2)
c1.metric("Mean Spread", f"{df_pair['spread'].mean():.4f}")
c2.metric("Std Dev of Spread", f"{df_pair['spread'].std():.4f}")

spread_fig = px.line(df_pair, x=df_pair.index, y="spread", title=f"{asset1} – {asset2} Spread", labels={"spread": "Spread", "index": "Date"})
spread_fig.add_hline(y=df_pair["spread"].mean(), line_dash="dash", line_color="red", annotation_text="Mean", annotation_position="bottom right")

z_fig = px.line(df_pair, x=df_pair.index, y="z_score", title="Z‑Score of Spread", labels={"z_score": "Z‑Score", "index": "Date"})
for lvl in [1, -1]:
    z_fig.add_hline(y=lvl, line_dash="dash", line_color="green", annotation_text=f"{lvl:+}", annotation_position="top right" if lvl > 0 else "bottom right")
z_fig.add_hline(y=0, line_dash="dot", line_color="black")

col1, col2 = st.columns(2, gap="large")
col1.plotly_chart(spread_fig, use_container_width=True)
col2.plotly_chart(z_fig, use_container_width=True)

st.subheader("Pair Summary")
st.dataframe(
    pairs_df[pairs_df["Asset1"] == asset1][
        [
            "Asset1",
            "Asset2",
            "P-Value",
            "Hurst",
            "Spread_Std",
            "P-Value_adj",
            "Hurst_adj",
            "Quality_Score",
        ]
    ].style.format(
        {
            "P-Value": "{:.5}",
            "P-Value_adj": "{:.5}",
            "Hurst": "{:.3f}",
            "Hurst_adj": "{:.3f}",
            "Spread_Std": "{:.4f}",
            "Quality_Score": "{:.2f}",
        }
    ),
    use_container_width=True,
)

st.subheader("All Pairs Summary")
st.dataframe(
    pairs_df[
        [
            "Asset1",
            "Asset2",
            "P-Value",
            "Hurst",
            "Spread_Std",
            "P-Value_adj",
            "Hurst_adj",
            "Quality_Score",
        ]
    ].style.format(
        {
            "P-Value": "{:.5}",
            "P-Value_adj": "{:.5}",
            "Hurst": "{:.3f}",
            "Hurst_adj": "{:.3f}",
            "Spread_Std": "{:.4f}",
            "Quality_Score": "{:.2f}",
        }
    ),
    use_container_width=True,
)
