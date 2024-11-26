# Serves as the main script to run the DRL algorithm
import pandas as pd
import numpy as np
import time
import os

from preprocessing.preprocessors import *
from config import config
from model.transfer_models import run_ensemble_strategy
import logging
import traceback

def run_model(use_turbulence=True):
    # read and preprocess data
    preprocessed_path = f"done_data_{config.dj_start_date}_{config.dj_end_date}.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = retrieve_DJ30_data()
        data = preprocess_data(data)
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    # For validation purpose, plot turbulence index
    # plot_turbulence_index(data)

    unique_trade_date = data[(data.datadate > config.trade_start_date)&(data.datadate <= config.trade_end_date)].datadate.unique()
    logging.info(f"Trade date count: {unique_trade_date}")

    try:
        run_ensemble_strategy(df = data,
                            unique_trade_date = unique_trade_date,
                            rebalance_window = config.rebalance_window,
                            validation_window = config.validation_window,
                            global_seed = 42,
                            use_turbulence=use_turbulence,
                            early_stopping=False)
    except Exception as e:
        logging.error("Error type: %s", type(e).__name__)
        logging.error("Error message: %s", str(e))
        logging.error(traceback.format_exc())  # Optionally logs the full stack trace

if __name__ == "__main__":
    run_model()