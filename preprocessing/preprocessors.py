from unittest.mock import inplace

import numpy as np
import pandas as pd
from pkg_resources import non_empty_lines
from stockstats import StockDataFrame as Sdf
from config import config
import yfinance as yf
import os
import matplotlib.pyplot as plt

def data_split(df, start, end):
    """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data = data.sort_values(['datadate', 'tic'], ignore_index=True)
    # data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

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
    add turbulence index for the dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    df = data.copy()
    df_price_pivot = df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()

    # start after a year, take the first year as a historical data calculation base
    start = 252
    turbulence_index = [0] * start
    # turbulence_index = [0]

    count = 0
    for i in range(start, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]

        # Expanding window to calculate the historical turbulence index
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index]]
        cov_temp = hist_price.cov()
        current_temp = (current_price - np.mean(hist_price, axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)

    turbulence_index = pd.DataFrame({'datadate': df_price_pivot.index,
                                     'turbulence': turbulence_index})

    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate', 'tic']).reset_index(drop=True)

    return df

def plot_turbulence_index(data):
    '''
    This function plot the turbulence index for validation
    :param data: pandas dataframe
    :return: None
    '''
    df = data.copy()
    df = df.groupby('datadate').agg({'turbulence': 'first'}).reset_index()
    df['datadate'] = pd.to_datetime(df['datadate'], format='%Y%m%d')
    plt.figure(figsize=(10, 6))
    plt.plot(df['datadate'], df['turbulence'], color='blue', label='Turbulence Index')

    plt.title('Turbulence Index Over Time')
    plt.xlabel('Date')
    plt.ylabel('Turbulence Index')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


## For testing purposes
# if __name__ == "__main__":
#     retrieve_DJ30_data()