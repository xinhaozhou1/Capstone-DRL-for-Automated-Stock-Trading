from unittest.mock import inplace

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config
import yfinance as yf
import os

def retrieve_DJ30_data():
    """
    Retrieve DJIA data according to the configuration
    :return: (df) pandas dataframe
    """
    tickers = config.dj_stock_ticker
    start_date = config.dj_start_date
    end_date = config.dj_end_date

    # Retrieve the data from Yahoo Finance
    data = pd.DataFrame()
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data['tic'] = ticker
        stock_data['datadate'] = stock_data.index
        stock_data['datadate'] = stock_data['datadate'].apply(lambda x: int(x.strftime('%Y%m%d')))
        stock_data = stock_data[['datadate', 'tic', 'Adj Close', 'Open', 'High', 'Low', 'Volume']]
        stock_data.columns = ['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']
        data = pd.concat([data, stock_data])

    # Save the data to the specified path
    data.to_csv(config.dj_data_path, index=False)
    return data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    df_preprocess = data.copy()
    df_preprocess = df_preprocess.sort_values(['tic', 'datadate'], ignore_index=True)
    df_final = add_technical_indicator(df_preprocess)
    df_final.fillna(method = 'bfill', inplace = True)
    return df_final

def add_technical_indicator(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the data
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(data.copy())
    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    adx = pd.DataFrame()

    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_adx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_adx = pd.DataFrame(temp_adx)
        adx = adx.append(temp_adx, ignore_index=True)

    data['macd'] = macd
    data['rsi'] = rsi
    data['cci'] = cci
    data['adx'] = adx
    return data

def add_turbulence(data: pd.DataFrame):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    # TODO: Implement this function
    return NotImplementedError

## For testing purposes
# if __name__ == "__main__":
#     retrieve_DJ30_data()