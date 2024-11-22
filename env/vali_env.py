import numpy as np
import pandas as pd
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

from env.train_env import StockEnvTrain  # import parent class
from config import config

""" Parameters """
NUM_STOCK = 30
INITIAL_ACCOUNT_BALANCE = 1000000
NUM_SHARES_PER_TRADE = 100
TRANSACTION_FEE_PERCENT = 0.001
STATE_SHAPE = 1 + NUM_STOCK * 6
# turbulence index: 90-150 reasonable threshold
TURBULENCE_THRESHOLD = 140

class StockEnvValidation(StockEnvTrain):
    """A stock trading validation environment inheriting from StockEnvTrain"""

    def __init__(self, df, day=0, turbulence_threshold=TURBULENCE_THRESHOLD, iteration='', seed=42):
        super().__init__(df, day, seed)
        self.turbulence_threshold = turbulence_threshold
        self.turbulence = 0
        self.iteration = iteration
    
    def step(self, actions):
        # Check if the environment is terminal
        self.is_terminal = self.day >= len(self.df) - 1
        
        if self.is_terminal:
            # return super().step(actions)
            # plt.plot(self.asset_memory, 'r')
            # plt.savefig(f'{config.results_dir}/account_value_validation_{self.iteration}.png')
            # plt.close()

            df_total_value = pd.DataFrame({
                    'datadate': [entry[0] for entry in self.asset_memory],
                    'account_value': [entry[1] for entry in self.asset_memory]
                })
            df_total_value.to_csv(f'{config.results_dir}/account_value_validation_{self.iteration}.csv', index=False)

            end_total_asset = self._get_asset_value_from_state()
            logging.info("Terminal Asset Value: {}".format(end_total_asset))

            df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            logging.info(f"Sharpe Ratio: {sharpe}")

            return self.state, self.reward, self.is_terminal, {}

        else:
            begin_total_asset = self._get_asset_value_from_state()

            actions = (actions * NUM_SHARES_PER_TRADE).astype(int)
            if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-NUM_SHARES_PER_TRADE] * NUM_STOCK)
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
            timestamp = self.data['datadate']
            self.turbulence = self.data['turbulence'][0]
            self.state = np.array([cash_balance] + list(self.data.adjcp) + list(stock_shares) + \
                list(self.data.macd) + list(self.data.rsi) + list(self.data.cci) + list(self.data.adx))
            assert self.state.shape == (STATE_SHAPE,)

            end_total_asset = self._get_asset_value_from_state()

            # Original reward
            # self.reward = end_total_asset - begin_total_asset

            # Cumulative reward 1
            # current_trade_return = (end_total_asset - begin_total_asset) / begin_total_asset
            # cumulative_trade_return = (end_total_asset / INITIAL_ACCOUNT_BALANCE) ** (1/self.day) - 1
            # decay_rate = 0.2
            # self.reward = (1-decay_rate) * current_trade_return + decay_rate * cumulative_trade_return

            # Cumulative reward 2
            current_trade_return = (end_total_asset - begin_total_asset) / begin_total_asset
            prev_reward = self.rewards_memory[-1][1] if self.rewards_memory else 0
            decay_rate = 0.2
            self.reward = current_trade_return + decay_rate * prev_reward
            
            self.rewards_memory.append((timestamp, self.reward))
            self.asset_memory.append((timestamp, end_total_asset))

        return self.state, self.reward, self.is_terminal, {}
    
    def reset(self):
        self.state = super().reset()
        self.turbulence = 0
        return self.state
    
    def _buy_stock(self, index, action):
        """ Override buy_stock to handle turbulence """
        # Only buy stock if turbulence is below the threshold
        if self._get_turbulence() < self.turbulence_threshold:
            # Call the parent's buy stock logic
            super()._buy_stock(index, action)
        else:
            # If turbulence is above the threshold, do nothing
            pass

    def _sell_stock(self, index, action):
        """ Override sell_stock to handle turbulence """
        # Only sell stock if turbulence is below the threshold
        if self._get_turbulence() < self.turbulence_threshold:
            # Call the parent's sell stock logic
            super()._sell_stock(index, action)
        else:
            # if turbulence goes over threshold, just clear out all positions 
            curr_stock_share = self.state[1 + NUM_STOCK + index]
            if curr_stock_share > 0:
                stock_price = self.state[1 + index]
                num_share_to_sell = curr_stock_share
                # Update cash balance
                self.state[0] += stock_price * num_share_to_sell * (1 - TRANSACTION_FEE_PERCENT)
                # Update number of shares
                self.state[1 + NUM_STOCK + index] = 0
                # Update cost and trades
                self.cost += stock_price * num_share_to_sell * TRANSACTION_FEE_PERCENT
                self.trades += 1
            else:
                pass

    def _get_turbulence(self):
        return self.turbulence
