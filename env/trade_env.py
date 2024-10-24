import pandas as pd
import numpy as np
from env.vali_env import StockEnvValidation  # import parent class
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

class StockEnvTrade(StockEnvValidation):
    def __init__(self, df, day=0, turbulence_threshold=140, is_initial=True, previous_state=[], model_name='', iteration='', seed=42):
        super().__init__(df, day, turbulence_threshold=turbulence_threshold, iteration=iteration, seed=seed)
        self.is_initial = is_initial
        self.previous_state = np.array(previous_state).reshape(-1)
        assert self.state.shape == (STATE_SHAPE,)
        self.model_name = model_name        

    def step(self, actions):
        # Check if the environment is terminal
        self.is_terminal = self.day >= len(self.df) - 1

        if self.is_terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig(f'{config.results_dir}/account_value_trade_{self.model_name}_{self.iteration}.png')
            plt.close()

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(f'{config.results_dir}/account_value_trade_{self.model_name}_{self.iteration}.csv')
            
            end_total_asset = self._get_asset_value_from_state()
            logging.info("Previous Total Asset: {}".format(self.asset_memory[0]))
            logging.info("Terminal Asset Value: {}".format(end_total_asset))
            logging.info("Total Reward: {}".format(end_total_asset - self.asset_memory[0]))
            logging.info(f"Total Cost: {self.cost}")
            logging.info(f"Total Trades: {self.trades}")

            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            logging.info(f"Sharpe: {sharpe}")
            
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv(f'{config.results_dir}/account_rewards_trade_{self.model_name}_{self.iteration}.csv')
            
            return self.state, self.reward, self.is_terminal, {}

        else:
            actions = (actions * NUM_SHARES_PER_TRADE).astype(int)
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-NUM_SHARES_PER_TRADE] * NUM_STOCK)

            begin_total_asset = self._get_asset_value_from_state()
            
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])
            
            cash_balance = self.state[0]
            stock_shares = self.state[(NUM_STOCK + 1) : (NUM_STOCK * 2 + 1)]

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            self.turbulence = self.data['turbulence'][0]
            assert type(self.turbulence) == float
            self.state = np.array([cash_balance] + list(self.data.adjcp) + list(stock_shares) + \
                list(self.data.macd) + list(self.data.rsi) + list(self.data.cci) + list(self.data.adx))
            assert self.state.shape == (STATE_SHAPE,)
            
            end_total_asset = self._get_asset_value_from_state()

            self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)

        return self.state, self.reward, self.is_terminal, {}
    
    def reset(self):  
        if self.is_initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.is_terminal = False 
            self.rewards_memory = []
            self.state = np.array([INITIAL_ACCOUNT_BALANCE] + list(self.data.adjcp) + [0] * NUM_STOCK + \
                                  list(self.data.macd) + list(self.data.rsi) + list(self.data.cci) + list(self.data.adx))
            assert self.state.shape == (STATE_SHAPE,)
        else:
            previous_total_asset = self._get_asset_value_from_prev_state()
            self.asset_memory = [previous_total_asset]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.is_terminal = False 
            self.rewards_memory = []

            prev_cash_balance = self.previous_state[0]
            prev_stock_shares = self.previous_state[(NUM_STOCK + 1) : (NUM_STOCK * 2 + 1)]

            self.state = np.array([prev_cash_balance] + list(self.data.adjcp) + list(prev_stock_shares) + \
                list(self.data.macd) + list(self.data.rsi) + list(self.data.cci) + list(self.data.adx))
            assert self.state.shape == (STATE_SHAPE,)
        return self.state

    def _get_asset_value_from_prev_state(self):
        prev_cash_balance = self.previous_state[0]
        prev_stock_prices = self.previous_state[1 : (NUM_STOCK + 1)]
        prev_stock_shares = self.previous_state[(NUM_STOCK + 1) : (NUM_STOCK * 2 + 1)]
        return prev_cash_balance + sum(np.array(prev_stock_prices) * np.array(prev_stock_shares))
    
    def render(self, mode = 'human', close = False):
        return self.state