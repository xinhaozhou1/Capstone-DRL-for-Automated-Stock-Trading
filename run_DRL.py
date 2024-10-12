# Serves as the main script to run the DRL algorithm
import pandas as pd
import numpy as np
import time
import os

from preprocessing.preprocessors import *
from config import config
from model.models import *

def run_model():
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
    plot_turbulence_index(data)
    # print(data.head())
    # print(data.size())

    unique_trade_date = data[(data.datadate > config.trade_start_date)&(data.datadate <= config.trade_end_date)].datadate.unique()
    print("Trade date count: ", unique_trade_date)

    run_ensemble_strategy(df = data,
                          unique_trade_date=unique_trade_date,
                          rebalance_window = config.rebalance_window,
                          validation_window = config.validation_window)


if __name__ == "__main__":
    run_model()