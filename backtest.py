import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO
import yfinance as yf

import pyfolio as pf
from stable_baselines3.common.vec_env import DummyVecEnv

from config.config import trade_end_date
from env.train_env import StockEnvTrain
from env.trade_env import StockEnvTrade
from model.models import *
from config import config
from preprocessing.preprocessors import *
import run_DRL

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

config.dj_start_date = '2009-01-01'
config.dj_end_date = '2024-10-31'
config.init_turbulence_sample_start_date = 20090000
config.init_turbulence_sample_end_date = 20191000
config.trade_start_date = 20200101
config.trade_end_date = 20241031

# debug settings
# TODO: Debug this
config.results_dir = 'results/test'

def backtest_strat(df):
    strategy_ret= df.copy()
    strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'])
    strategy_ret.set_index('Date', drop = False, inplace = True)
    strategy_ret.index = pd.to_datetime(strategy_ret.index)
    del strategy_ret['Date']
    ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
    return ts


def train_and_test_agent(df, env_train, agent_class, model_name, timestep, trade_start_date, trade_end_date):
    # Train the agent
    # TODO: Fix this bug
    logging.info(f"======Individual Agent Training from 20090000 to {train_data_end_date}")
    agent = agent_class('MlpPolicy', env_train, verbose=0)
    agent.learn(total_timesteps=timestep)

    # Prepare trade environment
    trade = data_split(df, start=trade_start_date, end=trade_end_date)
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade,
                                                   turbulence_threshold=1e6,
                                                   is_initial=True,
                                                   previous_state=[],
                                                   model_name=model_name,
                                                   iteration=trade_end_date,
                                                   seed=42)])
    obs_trade = env_trade.reset()
    done = False
    last_state = []

    while not done:
        action, _states = agent.predict(obs_trade)
        obs_trade, rewards, done, info = env_trade.step(action)
    last_state = info[0]['terminal_observation']

    # Save last state
    pd.DataFrame({'last_state': list(last_state)}).to_csv(
        f'{config.results_dir}/last_state_{model_name}_{trade_end_date}.csv', index=False)

    # Load account value for the strategy
    df_model = pd.read_csv(f'{config.results_dir}/account_value_trade_{model_name}_{trade_end_date}.csv')
    df_model['Date'] = pd.to_datetime(df_model['datadate'], format='%Y%m%d')
    df_model['daily_return'] = df_model['account_value'].pct_change(1)
    df_model['name'] =  model_name
    return df_model


def plot_performance(df_strats, df_account_value, df_dji):
    plt.figure(figsize=(10, 5))
    plt.plot(df_account_value['datadate'], df_account_value['cumret'], linestyle='-', color='r', label = 'Ensemble')
    plt.plot(df_dji['Date'], df_dji['cumret'], linestyle='-', color='b', label='DJI')
    for i in df_strats:
        plt.plot(i['Date'], i['cumret'], linestyle='-', label=i['name'].unique())
    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.title('Account Value Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{config.results_dir}/backtest.png')
    plt.close()

def backtest():
    # run_DRL.run_model()
    df = pd.read_csv(f"done_data_{config.dj_start_date}_{config.dj_end_date}.csv")
    unique_trade_date = df[(df.datadate > config.trade_start_date)
                           & (df.datadate <= config.trade_end_date)].datadate.unique()

    # Retrieve the account value for the ensemble strategy
    df_account_value = pd.DataFrame()
    for i in range(config.rebalance_window + config.validation_window, len(unique_trade_date) + 1, config.rebalance_window):
        temp = pd.read_csv(
            f'{config.results_dir}/account_value_trade_ensemble_{unique_trade_date[i]}.csv')
        df_account_value = df_account_value.append(temp, ignore_index=True)

    df_account_value['datadate'] = pd.to_datetime(df_account_value['datadate'], format='%Y%m%d')
    df_account_value['daily_return'] = df_account_value['account_value'].pct_change(1)

    # Import DJI Index Data
    ticker = "^DJI"
    start_date = df_account_value['datadate'].min().strftime('%Y-%m-%d')
    end_date = df_account_value['datadate'].max().strftime('%Y-%m-%d')

    dji_data = yf.download(ticker, start=start_date, end=end_date)
    dji_data['daily_return'] = dji_data['Adj Close'].pct_change(1)
    dji_data.reset_index(inplace=True)

    # Retrieve individual agents' performance
    train_start_date = config.init_turbulence_sample_start_date
    train_end_date = unique_trade_date[0]
    trade_end_date = config.trade_end_date

    train = data_split(df, start=train_start_date, end=train_end_date)
    env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

    # TODO: Add log file
    df_a2c = train_and_test_agent(df, env_train, A2C, "A2C", 30000, train_end_date, trade_end_date)
    df_ddpg = train_and_test_agent(df, env_train, DDPG, "DDPG", 5000, train_end_date, trade_end_date)
    df_ppo = train_and_test_agent(df, env_train, PPO, "PPO", 80000, train_end_date, trade_end_date)
    agents = [df_a2c, df_ddpg, df_ppo]
    for i in agents:
        i['cumret'] = (1 + i['daily_return']).cumprod() - 1
    plot_performance(agents, df_account_value, dji_data)


    # Compare the two strategies
    df_account_value.rename(columns={'datadate':'Date'}, inplace=True)
    ensemble_strat = backtest_strat(df_account_value)
    dow_strat = backtest_strat(dji_data)

    # Generate a backtest tear sheet
    pdf_path = f"{config.results_dir}/backtest_tear_sheet.pdf"
    with PdfPages(pdf_path) as pdf:
        with pf.plotting.plotting_context(font_scale=1.1):
            pf.create_full_tear_sheet(returns=ensemble_strat, benchmark_rets=dow_strat, set_context=False)
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    backtest()