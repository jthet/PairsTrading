import os, sys
from pathlib import Path
import logging
import streamlit as st
import pandas as pd
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.PairsFinder import PairsFinder as StrategyClass

st.set_page_config(page_title="Pairs Finder", layout="wide")
st.title("🔍 Pairs Finder")

default_url = "https://raw.githubusercontent.com/jthet/PairsTrading/refs/heads/main/data/russel_data_full.csv"
st.sidebar.header("Data Source")
data_url = st.sidebar.text_input("Price CSV URL (raw GitHub link)", default_url)
ticker_file = st.sidebar.file_uploader("Or upload tickers CSV", type=["csv"])

st.sidebar.header("Pipeline Parameters")
start = st.sidebar.date_input("Start date", value=pd.to_datetime("2019-01-01"))
end = st.sidebar.date_input("End date", value=pd.to_datetime("2025-01-01"))
split_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.1, 0.9, 0.5, 0.05)
cointegration_threshold = st.sidebar.number_input(
    "Cointegration p-value threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
hurst_threshold = st.sidebar.number_input(
    "Hurst exponent threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
st.sidebar.header("Clustering Params")
num_clusters = st.sidebar.number_input(
    "Number of clusters (0 for auto)", min_value=0, value=0, step=1)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)
st.sidebar.header("Autoencoder Params")
encoding_dim = st.sidebar.number_input("Encoding dimension", min_value=1, value=5, step=1)
hidden_dim = st.sidebar.number_input("Hidden dimension", min_value=1, value=64, step=1)
num_epochs = st.sidebar.number_input("Epochs", min_value=1, value=500, step=1)
lr = st.sidebar.number_input("Learning rate", min_value=0.0001, value=0.01, format="%.4f")
print_every = st.sidebar.number_input("Log every n epochs", min_value=1, value=50, step=1)

run_button = st.sidebar.button("Run Pairs Finder")

csv_display = st.empty()
log_container = st.empty()
output_area = st.empty()

class StreamlitTerminalHandler(logging.Handler):
    def __init__(self, container, max_lines=20):
        super().__init__()
        self.container = container      # this is your st.empty()
        self.logs = []
        self.max_lines = max_lines
        self.setLevel(logging.INFO)

    def emit(self, record):
        # 1) collect & trim
        self.logs.append(self.format(record))
        if len(self.logs) > self.max_lines:
            self.logs = self.logs[-self.max_lines:]
        # 2) overwrite the same placeholder with a code block
        self.container.code("\n".join(self.logs))


if run_button:
    handler = StreamlitTerminalHandler(log_container)
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    logger = logging.getLogger('backend.PairsFinder')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    ae_params = {
        'encoding_dim': encoding_dim,
        'hidden_dim': hidden_dim,
        'num_epochs': num_epochs,
        'lr': lr,
        'print_every': print_every
    }
    clust_params = {
        'num_clusters': None if num_clusters == 0 else num_clusters,
        'random_state': random_state
    }

    try:
        if data_url:
            logger.info("Loading price data from URL...")
            full_df = pd.read_csv(data_url, parse_dates=['Date'], index_col='Date')
            logger.info(f"Loaded price data: {full_df.shape[0]} rows & {full_df.shape[1]} assets.")
            pf = StrategyClass(
                tickers=full_df.columns.tolist(),
                start=str(full_df.index.min().date()),
                end=str(full_df.index.max().date()),
                split_ratio=split_ratio,
                cointegration_threshold=cointegration_threshold,
                hurst_threshold=hurst_threshold,
                autoencoder_params=ae_params,
                clustering_params=clust_params
            )
            pf.full_data = full_df
            pairs_df = pf.run_pipeline()
            
        elif ticker_file:
            tickers_df = pd.read_csv(ticker_file)
            tickers = tickers_df.iloc[:,0].dropna().astype(str).tolist()
            logger.info("Running pipeline with uploaded tickers...")
            pf = StrategyClass(
                tickers=tickers,
                start=str(start),
                end=str(end),
                split_ratio=split_ratio,
                cointegration_threshold=cointegration_threshold,
                hurst_threshold=hurst_threshold,
                autoencoder_params=ae_params,
                clustering_params=clust_params
            )
            pairs_df = pf.run_pipeline()
        else:
            raise ValueError("Provide either a price CSV URL or upload tickers CSV.")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        pairs_df = pd.DataFrame()

    if not pairs_df.empty:
        st.session_state['full_df'] = full_df
        st.session_state['pairs_df'] = pairs_df
        st.session_state['train_data'] = pf.train_data
        st.session_state['test_data'] = pf.test_data

        csv_display.subheader("Cointegrated Pairs")
        csv_display.dataframe(pairs_df)
        csv_display.download_button("Download Pairs CSV", pairs_df.to_csv(index=False).encode(), "pairs_df.csv", "text/csv")


        # st.subheader("Train Data Sample")
        # st.dataframe(pf.train_data.head())
        # st.download_button("Download Train CSV", pf.train_data.to_csv().encode(), "russel_data_train.csv", "text/csv")

        # st.subheader("Test Data Sample")
        # st.dataframe(pf.test_data.head())
        # st.download_button("Download Test CSV", pf.test_data.to_csv().encode(), "russel_data_test.csv", "text/csv")
    if pairs_df.empty:
        output_area.warning("No cointegrated pairs found or pipeline failed. Check logs above.")
    else:
        output_area.success("Done! Pairs, train, and test data saved to session.")
        st.subheader("Cointegrated Pairs")
        st.dataframe(pairs_df)
