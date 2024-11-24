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
            # print(self.asset_memory)
            # plt.plot(self.asset_memory,'r')
            datadates, account_values = zip(*self.asset_memory)
            datadates = pd.to_datetime(datadates, format='%Y%m%d')
            plt.plot(datadates, account_values, 'r')
            plt.xlabel('Date')
            plt.ylabel('Account Value')
            plt.title(f'Account Value from {datadates[0].strftime("%Y-%m-%d")} to {datadates[-1].strftime("%Y-%m-%d")}')
            plt.xticks(rotation=45)
            plt.savefig(f'{config.results_dir}/account_value_trade_{self.model_name}_{self.iteration}.png')
            plt.close()

            df_total_value = pd.DataFrame({
                    'datadate': [entry[0] for entry in self.asset_memory],
                    'account_value': [entry[1] for entry in self.asset_memory]
                })
            df_total_value.to_csv(f'{config.results_dir}/account_value_trade_{self.model_name}_{self.iteration}.csv', index=False)
            
            end_total_asset = self._get_asset_value_from_state()
            logging.info("Previous Total Asset: {}".format(self.asset_memory[0][1]))
            logging.info("Terminal Asset Value: {}".format(end_total_asset))
            logging.info("Total Reward: {}".format(end_total_asset - self.asset_memory[0][1]))
            logging.info(f"Total Cost: {self.cost}")
            logging.info(f"Total Trades: {self.trades}")

            df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            logging.info(f"Sharpe: {sharpe}")
            
            df_rewards = pd.DataFrame({
                'datadate': [entry[0] for entry in self.rewards_memory],
                'reward': [entry[1] for entry in self.rewards_memory]})
            df_rewards.to_csv(f'{config.results_dir}/account_rewards_trade_{self.model_name}_{self.iteration}.csv', index=False)
            
            return self.state, self.reward, self.is_terminal, {}

        else:
            actions = (actions * NUM_SHARES_PER_TRADE).astype(int)
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-NUM_SHARES_PER_TRADE] * NUM_STOCK)

            begin_total_asset = self._get_asset_value_from_state()
            
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            # freeze_state = self._freeze_state()

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])
            
            cash_balance = self.state[0]
            stock_shares = self.state[(NUM_STOCK + 1) : (NUM_STOCK * 2 + 1)]

            self.day += 1
            self.data = self.df.loc[self.day,:]
            timestamp = self.data['datadate']
            self.turbulence = self.data['turbulence'][0]
            assert type(self.turbulence) == float
            self.state = np.array([cash_balance] + list(self.data.adjcp) + list(stock_shares) + \
                list(self.data.macd) + list(self.data.rsi) + list(self.data.cci) + list(self.data.adx))
            assert self.state.shape == (STATE_SHAPE,)
            
            end_total_asset = self._get_asset_value_from_state()

            self.reward = self.reward_function(begin_total_asset, end_total_asset, reward_type=2, decay_rate=0.2)
            self.rewards_memory.append((timestamp, self.reward))

            # current_trade_return = (end_total_asset - begin_total_asset) / begin_total_asset
            # self.returns_history.append(current_trade_return)

            # if self.stop_trade:
            #     self.state = freeze_state
            #     end_total_asset = self._get_asset_value_from_state()
            
            self.asset_memory.append((timestamp, end_total_asset))

            # window_cumu_return = self._get_cumulative_returns(window=3)
            # if window_cumu_return < -0.02 and not self.stop_trade:
            #     logging.info(f"Trade frozen at cumulative return = {window_cumu_return}")
            #     self.stop_trade = True
            # elif window_cumu_return > 0.01 and self.stop_trade:
            #     logging.info(f"Trading resumed at cumulative return = {window_cumu_return}")
            #     self.stop_trade = False

        return self.state, self.reward, self.is_terminal, {}
    
    def reset(self):
        if self.is_initial:
            self.asset_memory = [(self.df.iloc[0]['datadate'], INITIAL_ACCOUNT_BALANCE)]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.is_terminal = False 
            self.rewards_memory = []
            self.state = np.array([INITIAL_ACCOUNT_BALANCE] + list(self.data.adjcp) + [0] * NUM_STOCK + \
                                  list(self.data.macd) + list(self.data.rsi) + list(self.data.cci) + list(self.data.adx))
            self.stop_trade = False
            self.returns_history = []
            assert self.state.shape == (STATE_SHAPE,)
        else:
            previous_total_asset = self._get_asset_value_from_prev_state()
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.asset_memory = [(self.data['datadate'], previous_total_asset)]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.is_terminal = False 
            self.rewards_memory = []

            prev_cash_balance = self.previous_state[0]
            prev_stock_shares = self.previous_state[(NUM_STOCK + 1) : (NUM_STOCK * 2 + 1)]

            self.state = np.array([prev_cash_balance] + list(self.data.adjcp) + list(prev_stock_shares) + \
                list(self.data.macd) + list(self.data.rsi) + list(self.data.cci) + list(self.data.adx))
            self.stop_trade = False
            self.returns_history = []
            assert self.state.shape == (STATE_SHAPE,)
        return self.state

    def _get_asset_value_from_prev_state(self):
        prev_cash_balance = self.previous_state[0]
        prev_stock_prices = self.previous_state[1 : (NUM_STOCK + 1)]
        prev_stock_shares = self.previous_state[(NUM_STOCK + 1) : (NUM_STOCK * 2 + 1)]
        return prev_cash_balance + sum(np.array(prev_stock_prices) * np.array(prev_stock_shares))
    
    def render(self, mode = 'human', close = False):
        return self.state
    
    def reward_function(self, begin_total_asset, end_total_asset, reward_type=2, decay_rate=0.2):
        if reward_type == 0:
            # Original reward
            return end_total_asset - begin_total_asset
        elif reward_type == 1:
            # Cumulative reward 1
            current_trade_return = (end_total_asset - begin_total_asset) / begin_total_asset
            previous_total_asset = self.asset_memory[0][1]
            cumulative_trade_return = (end_total_asset / previous_total_asset) ** (1/self.day) - 1
            return (1-decay_rate) * current_trade_return + decay_rate * cumulative_trade_return
        elif reward_type == 2:
            # Cumulative reward 2
            current_trade_return = (end_total_asset - begin_total_asset) / begin_total_asset
            prev_reward = self.rewards_memory[-1][1] if self.rewards_memory else 0
            return current_trade_return + decay_rate * prev_reward
        
    def _get_cumulative_returns(self, window=5):
        returns_list = self.returns_history[-window:]
        cumulative_return = 1
        for r in returns_list:
            cumulative_return *= (1+r)
        return cumulative_return ** (1/len(returns_list)) - 1
    
    def _freeze_state(self):
        freeze_state = self.state.copy()
        if self.stop_trade:
            for i in range(NUM_STOCK):
                curr_stock_share = freeze_state[1 + NUM_STOCK + i]
                if curr_stock_share > 0:
                    stock_price = freeze_state[1 + i]
                    num_share_to_sell = curr_stock_share
                    # Update cash balance
                    freeze_state[0] += stock_price * num_share_to_sell * (1 - TRANSACTION_FEE_PERCENT)
                    # Update number of shares
                    freeze_state[1 + NUM_STOCK + i] = 0
        return freeze_state