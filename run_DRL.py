# Serves as the main script to run the DRL algorithm
import pandas as pd
import numpy as np
import time
import os

from preprocessing.preprocessors import *
from config.config import *

def run_model():
    # read and preprocess data
    preprocessed_path = "done_data.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = retrieve_DJ30_data()
        data = preprocess_data(data)

        # TODO: Implement add_turbulence
        #data = add_turbulence(data)

        data.to_csv(preprocessed_path)

    # print(data.head())
    # print(data.size())

    # TODO: Implement unique_trade_date and run_ensemble_strategy

if __name__ == "__main__":
    run_model()