import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from config import config
import logging

""" Parameters """
NUM_STOCK = 30
INITIAL_ACCOUNT_BALANCE = 1000000
NUM_SHARES_PER_TRADE = 100
TRANSACTION_FEE_PERCENT = 0.001
# Shape = [current balance] + [prices 1-30] + [owned shares 1-30] + [macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
STATE_SHAPE = 1 + NUM_STOCK * 6

class StockEnvTrain(gym.Env):
    """ A stock trading environment for OpenAI gym """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day = 0, seed=42):
        self.day = day
        self.df = self._sort_and_group_data(df)
        self.data = self.df.loc[self.day, :]
        self.is_terminal = False

        # Define action space and observation space
        self.action_space = spaces.Box(low = -1, high = 1,shape = (NUM_STOCK,))
        self.observation_space = spaces.Box(low = 0, high = np.inf, shape = (STATE_SHAPE,), dtype = np.float32)
        
        # Initialize state
        self.state = np.array([INITIAL_ACCOUNT_BALANCE] + list(self.data.adjcp) + [0] * NUM_STOCK + \
            list(self.data.macd) + list(self.data.rsi) + list(self.data.cci) + list(self.data.adx))
        
        assert self.state.shape == (STATE_SHAPE,)
        
        # Initialize other features
        self.reward = 0
        self.cost = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        self._seed(seed=seed)

    def step(self, actions):
        self.is_terminal = self.day >= len(self.df) - 1

        if self.is_terminal:
            end_total_asset = self._get_asset_value_from_state()
            logging.info("Terminal Asset Value: {}".format(end_total_asset))
            
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            logging.info(f"Sharpe Ratio: {sharpe}")
            
            return self.state, self.reward, self.is_terminal,{}

        else:
            begin_total_asset = self._get_asset_value_from_state()

            actions = (actions * NUM_SHARES_PER_TRADE).astype(int)
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])

            cash_balance = self.state[0]
            stock_shares = self.state[(NUM_STOCK + 1) : (NUM_STOCK * 2 + 1)]

            # Obtain next day stock prices and update state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.state = np.array([cash_balance] + list(self.data.adjcp) + list(stock_shares) + \
                list(self.data.macd) + list(self.data.rsi) + list(self.data.cci) + list(self.data.adx))
            assert self.state.shape == (STATE_SHAPE,)
            
            end_total_asset = self._get_asset_value_from_state()
                        
            self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)

        return self.state, self.reward, self.is_terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.is_terminal = False 
        self.rewards_memory = []
        self.state = np.array([INITIAL_ACCOUNT_BALANCE] + list(self.data.adjcp) + [0] * NUM_STOCK + \
            list(self.data.macd) + list(self.data.rsi) + list(self.data.cci) + list(self.data.adx))
        assert self.state.shape == (STATE_SHAPE,)
        return self.state
    
    def _sell_stock(self, index, action):
        curr_stock_share = self.state[1 + NUM_STOCK + index]
        if curr_stock_share > 0:
            stock_price = self.state[1 + index]
            num_share_to_sell = min(abs(action), curr_stock_share)
            # Update cash balance
            self.state[0] += stock_price * num_share_to_sell * (1 - TRANSACTION_FEE_PERCENT)
            # Update number of shares
            self.state[1 + NUM_STOCK + index] -= num_share_to_sell
            # Update cost and trades
            self.cost += stock_price * num_share_to_sell * TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            pass

    def _buy_stock(self, index, action):
        stock_price = self.state[1 + index]
        available_amount = self.state[0] // stock_price
        num_share_to_buy = min(action, available_amount)
        # Update cash balance
        self.state[0] -= stock_price * num_share_to_buy * (1 + TRANSACTION_FEE_PERCENT)
        # Update number of shares
        self.state[1 + NUM_STOCK + index] += num_share_to_buy
        # Update cost and trades
        self.cost += stock_price * num_share_to_buy * TRANSACTION_FEE_PERCENT
        self.trades += 1
    
    def _sort_and_group_data(self, df):
        """
        Data of same date grouped into tuple following the sequence of tickers as shown below:
        ('AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 
        'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT', 'XOM')
        """
        return df.sort_values(by=["datadate", "tic"]).groupby(by="datadate").agg(tuple).reset_index()
    
    def _get_asset_value_from_state(self):
        cash_balance = self.state[0]
        stock_prices = self.state[1 : (NUM_STOCK + 1)]
        stock_shares = self.state[(NUM_STOCK + 1) : (NUM_STOCK * 2 + 1)]
        return cash_balance + sum(np.array(stock_prices) * np.array(stock_shares))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(abs(int(seed)))
        return [seed]