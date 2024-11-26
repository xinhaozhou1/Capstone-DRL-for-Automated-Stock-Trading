import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC
import yfinance as yf

import pyfolio as pf
from stable_baselines3.common.vec_env import DummyVecEnv

from config.config import trade_end_date
from env.train_env import StockEnvTrain
from env.trade_env import StockEnvTrade
from config import config
from preprocessing.preprocessors import *
from run_DRL_transfer import run_model

import logging
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

config.dj_start_date = '2009-01-01'
config.dj_end_date = '2024-10-31'
config.init_turbulence_sample_start_date = 20090000
config.init_turbulence_sample_end_date = 20191000
config.trade_start_date = 20160101
config.trade_end_date = 20241031

def backtest_strat(df):
    strategy_ret= df.copy()
    strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'])
    strategy_ret.set_index('Date', drop = False, inplace = True)
    strategy_ret.index = pd.to_datetime(strategy_ret.index)
    del strategy_ret['Date']
    ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
    return ts


def train_and_test_agent(df, env_train, agent_class, model_name, timestep, train_start_date, train_end_date, trade_start_date, trade_end_date, is_initial, last_state):
    # Retrieve the trained agent of the corresponding period
    logging.info(f"======Individual Agent Trained Retrieved from {train_start_date} to {train_end_date}========")
    logging.info(f"Training model: {model_name}")
    # agent = agent_class('MlpPolicy', env_train, verbose=0)
    # agent.learn(total_timesteps=timestep)

    model_path = f"{config.trained_model_dir}/{model_name}_{trade_end_date}.zip"
    if os.path.exists(model_path):
        logging.info(f"Loading existing model: {model_name} from {model_path}")
        agent = agent_class.load(model_path, env=env_train)
    else:
        raise FileNotFoundError(f"Model {model_name} for period ending {trade_end_date} not found.")

    # Prepare trade environment
    logging.info(f"======{model_name} Individual Agent Trading from {trade_start_date} to {trade_end_date}======")
    trade = data_split(df, start=trade_start_date, end=trade_end_date)
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade,
                                                   turbulence_threshold=1e6,
                                                   is_initial=is_initial,
                                                   previous_state=last_state,
                                                   model_name=model_name,
                                                   iteration=trade_end_date,
                                                   seed=42)])
    obs_trade = env_trade.reset()
    done = False
    last_state = []

    while not done:
        action, _states = agent.predict(obs_trade, deterministic = True)
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
    return df_model, last_state


def plot_performance(df_agents, df_account_value, df_dji):
    plt.figure(figsize=(10, 5))
    plt.plot(df_account_value['datadate'], df_account_value['cumret'], linestyle='-', color='r', label = 'Ensemble')
    plt.plot(df_dji['Date'], df_dji['cumret'], linestyle='-', color='b', label='DJI')
    for name, group in df_agents.groupby('name'):
        plt.plot(group['Date'], group['cumret'], linestyle='-', label=name)

    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.title('Account Value Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{config.results_dir}/backtest.png')
    plt.close()

def backtest(use_turbulence=True):
    run_model(use_turbulence=use_turbulence)
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
    df_account_value['cumret'] = (1 + df_account_value['daily_return']).cumprod() - 1

    # Import DJI Index Data
    ticker = "^DJI"
    start_date = df_account_value['datadate'].min().strftime('%Y-%m-%d')
    end_date = df_account_value['datadate'].max().strftime('%Y-%m-%d')

    dji_data = yf.download(ticker, start=start_date, end=end_date)
    dji_data['daily_return'] = dji_data['Adj Close'].pct_change(1)
    dji_data['cumret'] = (1 + dji_data['daily_return']).cumprod() - 1
    dji_data.reset_index(inplace=True)

    # Retrieve individual agents' performance
    logging.info(f"======BackTest Session Starts from {int(start_date.replace('-', ''))} "
                 f"to {int(end_date.replace('-', ''))}======")
    agents = []

    # a2c = train_and_test_agent(df, env_train, A2C, "A2C", 30000, train_start_date, train_end_date, trade_start_date, trade_end_date)
    # ddpg = train_and_test_agent(df, env_train, DDPG, "DDPG", 5000, train_start_date, train_end_date, trade_start_date, trade_end_date)
    # ppo = train_and_test_agent(df, env_train, PPO, "PPO", 80000, train_start_date, train_end_date, trade_start_date,trade_end_date)
    # agents = [a2c, ddpg, ppo]

    # Implement rolling trading for agents
    a2c, ddpg, ppo, td3, sac = [pd.DataFrame(), []], [pd.DataFrame(), []], [pd.DataFrame(), []], [pd.DataFrame(), []], [pd.DataFrame(), []]
    df_agents = pd.DataFrame()
    for i in range(config.rebalance_window + config.validation_window, len(unique_trade_date), config.rebalance_window):
        train_start_date = config.init_turbulence_sample_start_date
        train_end_date = unique_trade_date[i - config.rebalance_window - config.validation_window]
        trade_start_date = unique_trade_date[i - config.rebalance_window]
        trade_end_date = unique_trade_date[i]
        is_initial = (i - config.rebalance_window - config.validation_window == 0)
        train = data_split(df, start=train_start_date, end=train_end_date)
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        a2c = train_and_test_agent(df, env_train, A2C, "A2C_30k_dow", config.A2C_ts, train_start_date, train_end_date,
                                      trade_start_date, trade_end_date, is_initial, last_state= a2c[1])
        ddpg = train_and_test_agent(df, env_train, DDPG, "DDPG_10k_dow", config.DDPG_ts, train_start_date, train_end_date,
                                       trade_start_date, trade_end_date, is_initial, last_state= ddpg[1])
        ppo = train_and_test_agent(df, env_train, PPO, "PPO_100k_dow", config.PPO_ts, train_start_date, train_end_date,
                                      trade_start_date, trade_end_date, is_initial, last_state= ppo[1])
        td3 = train_and_test_agent(df, env_train, TD3, "TD3_30k_dow", config.TD3_ts, train_start_date, train_end_date,
                                       trade_start_date, trade_end_date, is_initial, last_state= td3[1])
        sac = train_and_test_agent(df, env_train, SAC, "SAC_30k_dow", config.SAC_ts, train_start_date, train_end_date,
                                       trade_start_date, trade_end_date, is_initial, last_state= sac[1])
        df_agents = pd.concat([df_agents, a2c[0], ddpg[0], ppo[0], td3[0], sac[0]], axis=0)

    # Plot the performance of the agents
    df_agents = df_agents.sort_values(by='Date').reset_index(drop=True)
    df_agents['cumret'] = df_agents.groupby('name')['daily_return'].apply(lambda x: (1 + x).cumprod() - 1)
    plot_performance(df_agents, df_account_value, dji_data)

    # Compare the two strategies
    df_account_value.rename(columns={'datadate':'Date'}, inplace=True)
    ensemble_strat = backtest_strat(df_account_value)
    dow_strat = backtest_strat(dji_data)

    # Generate a backtest tear sheet
    # pdf_path = f"{config.results_dir}/backtest_tear_sheet.pdf"
    # with PdfPages(pdf_path) as pdf:
    #     with pf.plotting.plotting_context(font_scale=1.1):
    #         pf.create_full_tear_sheet(returns=ensemble_strat, benchmark_rets=dow_strat, set_context=False)

    #     for fig_num in plt.get_fignums():
    #         fig = plt.figure(fig_num)
    #         pdf.savefig(fig)
    #         plt.close(fig)

if __name__ == "__main__":
    backtest(use_turbulence=False)