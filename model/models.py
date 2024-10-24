from warnings import filterwarnings
filterwarnings(action="ignore")

import time
import gym
import numpy as np
import pandas as pd
import os
import logging

# Import models from stable baselines
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DDPG, PPO

from config import config
from preprocessing.preprocessors import *
import run_DRL

# Customized environment
from env.train_env import StockEnvTrain
from env.vali_env import StockEnvValidation
from env.trade_env import StockEnvTrade

def train_A2C(env_train, model_name, timesteps = 50000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose = 0)
    model.learn(total_timesteps = timesteps)
    end = time.time()

    model.save(f"{config.trained_model_dir}/{model_name}")
    logging.info(f'Training time (A2C): {(end - start) / 60} minutes')
    return model

def train_DDPG(env_train, model_name, timesteps = 50000):
    """DDPG model"""

    start = time.time()
    model = DDPG('MlpPolicy', env_train, verbose = 0)
    model.learn(total_timesteps = timesteps)
    end = time.time()

    model.save(f"{config.trained_model_dir}/{model_name}")
    logging.info(f'Training time (DDPG): {(end - start) / 60} minutes')
    return model

def train_PPO(env_train, model_name, timesteps = 50000):
    """PPO model"""

    start = time.time()
    model = PPO('MlpPolicy', env_train, verbose = 0)
    model.learn(total_timesteps = timesteps)
    end = time.time()

    model.save(f"{config.trained_model_dir}/{model_name}")
    logging.info(f'Training time (PPO): {(end - start) / 60} minutes')
    return model

def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)

def DRL_prediction(df, model, name, last_state, iter_num, unique_trade_date, rebalance_window, turbulence_threshold, is_initial):
    trade_data = data_split(df, 
                            start = unique_trade_date[iter_num - rebalance_window], 
                            end = unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data, 
                                                   turbulence_threshold = turbulence_threshold,
                                                   is_initial = is_initial,
                                                   previous_state = last_state,
                                                   model_name = name,
                                                   iteration = iter_num)])
    obs_trade = env_trade.reset()

    last_state = []
    for i in range(len(trade_data)):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data) - 2):
            last_state = obs_trade

    df_last_state = pd.DataFrame({'last_state': list(last_state)}, index=[0])
    df_last_state.to_csv(f'{config.results_dir}/last_state_{name}_{i}.csv', index=False)
    return last_state

def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv(f"{config.results_dir}/account_value_validation_{iteration}.csv", index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
    return sharpe


def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    logging.info("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state

    last_state_ensemble = []
    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []
    model_use = []

    insample_turbulence = df[(df.datadate < config.init_turbulence_sample_end_date)
                             & (df.datadate >= config.init_turbulence_sample_start_date)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    train_start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        logging.info("============================================")
        logging.info(f"Trade start date: {unique_trade_date[i]}")

        # Determine the initial state
        is_initial = False
        if i - rebalance_window - validation_window == 0:
            is_initial = True

        end_date_index = \
            df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window * 30 + 1

        # Consider the turbulence data for the current window
        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        logging.info(f"turbulence_threshold: {turbulence_threshold}")

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df,
                           start=config.init_turbulence_sample_start_date,
                           end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
        validation = data_split(df, 
                                start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation, turbulence_threshold=turbulence_threshold, iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        logging.info(f"======Model training from: 20090000 to {unique_trade_date[i - rebalance_window - validation_window]}")
        
        logging.info("======A2C Training========")
        model_a2c = train_A2C(env_train, model_name = "A2C_30k_dow_{}".format(i), timesteps = 30000)
        logging.info(f"======A2C Validation from {unique_trade_date[i - rebalance_window - validation_window]} to {unique_trade_date[i - rebalance_window]}")
        DRL_validation(model = model_a2c, test_data = validation, test_env = env_val, test_obs = obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        logging.info(f"A2C Sharpe Ratio: {sharpe_a2c}")

        logging.info("======PPO Training========")
        model_ppo = train_PPO(env_train, model_name = "PPO_100k_dow_{}".format(i), timesteps = 80000)
        logging.info(f"======PPO Validation from {unique_trade_date[i - rebalance_window - validation_window]} to {unique_trade_date[i - rebalance_window]}")
        DRL_validation(model = model_ppo, test_data = validation, test_env = env_val, test_obs = obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        logging.info(f"PPO Sharpe Ratio: {sharpe_ppo}")

        logging.info("======DDPG Training========")
        model_ddpg = train_DDPG(env_train, model_name = "DDPG_10k_dow_{}".format(i), timesteps = 5000)
        logging.info(f"======DDPG Validation from {unique_trade_date[i - rebalance_window - validation_window]} to {unique_trade_date[i - rebalance_window]}")
        DRL_validation(model = model_ddpg, test_data = validation, test_env = env_val, test_obs = obs_val)
        sharpe_ddpg = get_validation_sharpe(i)
        logging.info(f"DDPG Sharpe Ratio: {sharpe_ddpg}")

        a2c_sharpe_list.append(sharpe_a2c)
        ppo_sharpe_list.append(sharpe_ppo)
        ddpg_sharpe_list.append(sharpe_ddpg)

        max_model_sharpe = np.max([sharpe_a2c, sharpe_ddpg, sharpe_ppo])
        if max_model_sharpe == sharpe_ppo:
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif max_model_sharpe == sharpe_a2c:
            model_ensemble = model_a2c
            model_use.append('A2C')
        else:
            model_ensemble = model_ddpg
            model_use.append('DDPG')

        ############## Training and Validation ends ##############
        ############## Trading starts ##############
        logging.info(f"======Trading from {unique_trade_date[i - rebalance_window]} to {unique_trade_date[i]}")
        logging.info(f"Used Model: {model_use[-1]}")

        last_state_ensemble = DRL_prediction(df = df, 
                                             model = model_ensemble, 
                                             name="ensemble",
                                             last_state = last_state_ensemble, 
                                             iter_num = i,
                                             unique_trade_date = unique_trade_date,
                                             rebalance_window = rebalance_window,
                                             turbulence_threshold = turbulence_threshold,
                                             is_initial = is_initial)
        assert len(last_state_ensemble) != 0
        ############## Trading ends ##############

    train_end = time.time()
    logging.info(f"Training time: {train_end - train_start} seconds")
