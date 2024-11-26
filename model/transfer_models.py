from warnings import filterwarnings
filterwarnings(action="ignore")

import time
import gym
import numpy as np
import pandas as pd
import os
import logging
import torch
from collections import deque

# Import models from stable baselines
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC
from stable_baselines3.common.callbacks import BaseCallback

from config import config
from preprocessing.preprocessors import *
import run_DRL

# Customized environment
from env.train_env import StockEnvTrain
from env.vali_env import StockEnvValidation
from env.trade_env import StockEnvTrade

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DynamicTransferCallback(BaseCallback):
    """Dynamic evaluation callback that uses a sliding window to monitor training performance"""
    def __init__(self, check_freq, window_size=1000, threshold=0.8):
        super().__init__(verbose=1)
        self.check_freq = check_freq  # How often to check performance
        self.window_size = window_size  # Size of sliding window
        self.threshold = threshold  # Performance degradation threshold
        self.rewards_history = deque(maxlen=window_size*2)  # Store reward history using deque
        self.is_improving = True
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        # Get current reward
        current_reward = self.model.env.get_attr('reward')[0]
        self.rewards_history.append(current_reward)
        
        # Only evaluate after collecting enough samples
        if len(self.rewards_history) < self.window_size * 2:
            return True
            
        if self.n_calls % self.check_freq == 0:
            # Calculate average rewards for recent and previous windows
            recent_rewards = list(self.rewards_history)[-self.window_size:]
            previous_rewards = list(self.rewards_history)[-2*self.window_size:-self.window_size]
            
            recent_mean = np.mean(recent_rewards)
            previous_mean = np.mean(previous_rewards)
            
            # Update best mean reward
            if recent_mean > self.best_mean_reward:
                self.best_mean_reward = recent_mean
                self.is_improving = True
            
            # Calculate relative performance
            relative_perf = recent_mean / previous_mean if previous_mean != 0 else 1.0
            
            logging.info(f"Recent mean reward: {recent_mean:.2f}")
            logging.info(f"Previous mean reward: {previous_mean:.2f}")
            logging.info(f"Relative performance: {relative_perf:.2%}")
            
            # Stop if performance degrades significantly
            if relative_perf < self.threshold and not self.is_improving:
                logging.info("Stopping training due to performance degradation")
                return False
                
            self.is_improving = recent_mean >= previous_mean
            
        return True

def train_A2C(env_train, model_name, timesteps=50000, previous_model=None, early_stopping=False, seed=42):
    """A2C model with dynamic transfer learning"""
    start = time.time()
    
    if previous_model is not None:
        model = A2C('MlpPolicy', env_train, verbose=0, seed=seed, device=device,
                    policy_kwargs={"net_arch": previous_model.policy.net_arch})
        model.policy.load_state_dict(previous_model.policy.state_dict())
        
        # Use dynamic transfer callback or fixed step length
        if early_stopping:
            callback = DynamicTransferCallback(check_freq=1000, window_size=1000, threshold=0.8)
            model.learning_rate = model.learning_rate * 0.5
            model.learn(total_timesteps=timesteps // 2, callback=callback)
        else:
            model.learning_rate = model.learning_rate * 0.5
            model.learn(total_timesteps=timesteps // 4)
    else:
        model = A2C('MlpPolicy', env_train, verbose=0, seed=seed, device=device)
        model.learn(total_timesteps=timesteps)
    
    end = time.time()
    model.save(f"{config.trained_model_dir}/{model_name}")
    logging.info(f'Training time (A2C): {(end - start) / 60:.2f} minutes')
    return model

def train_DDPG(env_train, model_name, timesteps=50000, previous_model=None, early_stopping=False, seed=42):
    """DDPG model with dynamic transfer learning"""
    start = time.time()
    
    if previous_model is not None:
        model = DDPG('MlpPolicy', env_train, verbose=0, seed=seed, device=device,
                     policy_kwargs={"net_arch": previous_model.policy.net_arch})
        model.policy.load_state_dict(previous_model.policy.state_dict())
        model.critic.load_state_dict(previous_model.critic.state_dict())
        
        if early_stopping:
            callback = DynamicTransferCallback(check_freq=1000, window_size=1000, threshold=0.8)
            model.learning_rate = model.learning_rate * 0.5
            model.learn(total_timesteps=timesteps // 2, callback=callback)
        else:
            model.learning_rate = model.learning_rate * 0.5
            model.learn(total_timesteps=timesteps // 4)
    else:
        model = DDPG('MlpPolicy', env_train, verbose=0, seed=seed, device=device)
        model.learn(total_timesteps=timesteps)
    
    end = time.time()
    model.save(f"{config.trained_model_dir}/{model_name}")
    logging.info(f'Training time (DDPG): {(end - start) / 60:.2f} minutes')
    return model

def train_PPO(env_train, model_name, timesteps=50000, previous_model=None, early_stopping=False, seed=42):
    """PPO model with dynamic transfer learning"""
    start = time.time()
    
    if previous_model is not None:
        model = PPO('MlpPolicy', env_train, verbose=0, seed=seed, device=device,
                    policy_kwargs={"net_arch": previous_model.policy.net_arch})
        model.policy.load_state_dict(previous_model.policy.state_dict())
        
        if early_stopping:
            callback = DynamicTransferCallback(check_freq=1000, window_size=1000, threshold=0.8)
            model.learning_rate = model.learning_rate * 0.5
            model.learn(total_timesteps=timesteps // 2, callback=callback)
        else:
            model.learning_rate = model.learning_rate * 0.5
            model.learn(total_timesteps=timesteps // 4)
    else:
        model = PPO('MlpPolicy', env_train, verbose=0, seed=seed, device=device)
        model.learn(total_timesteps=timesteps)
    
    end = time.time()
    model.save(f"{config.trained_model_dir}/{model_name}")
    logging.info(f'Training time (PPO): {(end - start) / 60:.2f} minutes')
    return model

def train_TD3(env_train, model_name, timesteps=50000, previous_model=None, early_stopping=False, seed=42):
    """TD3 model with dynamic transfer learning"""
    start = time.time()
    
    if previous_model is not None:
        model = TD3('MlpPolicy', env_train, verbose=0, seed=seed, device=device,
                    policy_kwargs={"net_arch": previous_model.policy.net_arch})
        model.policy.load_state_dict(previous_model.policy.state_dict())
        model.critic.load_state_dict(previous_model.critic.state_dict())
        
        if early_stopping:
            callback = DynamicTransferCallback(check_freq=1000, window_size=1000, threshold=0.8)
            model.learning_rate = model.learning_rate * 0.5
            model.learn(total_timesteps=timesteps // 2, callback=callback)
        else:
            model.learning_rate = model.learning_rate * 0.5
            model.learn(total_timesteps=timesteps // 4)
    else:
        model = TD3('MlpPolicy', env_train, verbose=0, seed=seed, device=device)
        model.learn(total_timesteps=timesteps)
    
    end = time.time()
    model.save(f"{config.trained_model_dir}/{model_name}")
    logging.info(f'Training time (TD3): {(end - start) / 60:.2f} minutes')
    return model

def train_SAC(env_train, model_name, timesteps=50000, previous_model=None, early_stopping=False, seed=42):
    """SAC model with dynamic transfer learning"""
    start = time.time()
    
    if previous_model is not None:
        model = SAC('MlpPolicy', env_train, verbose=0, seed=seed, device=device,
                    policy_kwargs={"net_arch": previous_model.policy.net_arch})
        model.policy.load_state_dict(previous_model.policy.state_dict())
        model.critic.load_state_dict(previous_model.critic.state_dict())
        
        if early_stopping:
            callback = DynamicTransferCallback(check_freq=1000, window_size=1000, threshold=0.8)
            model.learning_rate = model.learning_rate * 0.5
            model.learn(total_timesteps=timesteps // 2, callback=callback)
        else:
            model.learning_rate = model.learning_rate * 0.5
            model.learn(total_timesteps=timesteps // 4)
    else:
        model = SAC('MlpPolicy', env_train, verbose=0, seed=seed, device=device)
        model.learn(total_timesteps=timesteps)
    
    end = time.time()
    model.save(f"{config.trained_model_dir}/{model_name}")
    logging.info(f'Training time (SAC): {(end - start) / 60:.2f} minutes')
    return model

def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs, deterministic = True)
        test_obs, rewards, dones, info = test_env.step(action)

def DRL_prediction(df, model, name, last_state, iter_num, unique_trade_date, rebalance_window, turbulence_threshold, is_initial, seed):
    val_data_end_date = unique_trade_date[iter_num - rebalance_window]
    trade_data_end_date = unique_trade_date[iter_num]
    trade_data = data_split(df, start = val_data_end_date, end = trade_data_end_date)
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data, 
                                                   turbulence_threshold = turbulence_threshold,
                                                   is_initial = is_initial,
                                                   previous_state = last_state,
                                                   model_name = name,
                                                   iteration = trade_data_end_date,
                                                   seed=seed)])
    obs_trade = env_trade.reset()

    last_state = []
    done = False
    while not done:
        action, _states = model.predict(obs_trade, deterministic = True)
        obs_trade, rewards, done, info = env_trade.step(action)
    last_state = info[0]['terminal_observation']

    df_last_state = pd.DataFrame({'last_state': list(last_state)}, index = list(range(len(last_state))))
    df_last_state.to_csv(f'{config.results_dir}/last_state_{name}_{trade_data_end_date}.csv', index=False)
    return last_state

def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv(f"{config.results_dir}/account_value_validation_{iteration}.csv", index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
    return sharpe

def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window, global_seed=42, early_stopping=False, use_turbulence=True) -> None:
    """Ensemble Strategy that combines PPO, A2C, DDPG, TD3 and SAC"""
    logging.info("============Start Ensemble Strategy============")
    logging.info("AAAAA")
    
    last_state_ensemble = []
    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []
    td3_sharpe_list = []
    sac_sharpe_list = []
    model_use = []

    # Store previous models
    previous_models = {
        "A2C": None,
        "DDPG": None,
        "PPO": None,
        "TD3": None,
        "SAC": None
    }

    insample_turbulence = df[(df.datadate < config.init_turbulence_sample_end_date)
                             & (df.datadate >= config.init_turbulence_sample_start_date)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    
    if use_turbulence:
        insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)
    else:
        insample_turbulence_threshold = 1e6

    train_start = time.time()
    rng = np.random.default_rng(global_seed)
    num_iter = (len(unique_trade_date) - (rebalance_window + validation_window)) // rebalance_window + 1
    seeds = rng.integers(0, 10000, size=num_iter)
    seed_iter = 0
    
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        train_data_start_date = config.init_turbulence_sample_start_date
        train_data_end_date = unique_trade_date[i - rebalance_window - validation_window]
        val_data_end_date = unique_trade_date[i - rebalance_window]
        trade_data_end_date = unique_trade_date[i]

        logging.info("============================================")
        logging.info(f"Trading session until date: {trade_data_end_date}")
        
        seed = abs(int(seeds[seed_iter]))
        is_initial = (i - rebalance_window - validation_window == 0)

        end_date_index = df.index[df["datadate"] == train_data_end_date].to_list()[-1]
        start_date_index = end_date_index - validation_window * 30 + 1

        # Consider the turbulence data for the current window
        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if use_turbulence:
            if historical_turbulence_mean > insample_turbulence_threshold:
                turbulence_threshold = insample_turbulence_threshold
            else:
                turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        else:
            turbulence_threshold = 1e6
            
        logging.info(f"Turbulence Threshold: {turbulence_threshold}")

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=train_data_start_date, end=train_data_end_date)
        env_train = DummyVecEnv([lambda: StockEnvTrain(train, seed=seed)])

        ## validation env
        validation = data_split(df, start=train_data_end_date, end=val_data_end_date)
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation, 
                                                          turbulence_threshold=turbulence_threshold, 
                                                          iteration=trade_data_end_date, 
                                                          seed=seed)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        logging.info(f"======Model training from 20090000 to {train_data_end_date}")
        
        logging.info("======A2C Training========")
        model_a2c = train_A2C(env_train, f"A2C_30k_dow_{trade_data_end_date}", 
                             timesteps=config.A2C_ts, previous_model=previous_models["A2C"], 
                             seed=seed, early_stopping=early_stopping)
        previous_models["A2C"] = model_a2c        
        logging.info(f"======A2C Validation from {train_data_end_date} to {val_data_end_date}")
        DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(trade_data_end_date)
        logging.info(f"A2C Sharpe Ratio: {sharpe_a2c}")

        logging.info("======PPO Training========")
        model_ppo = train_PPO(env_train, f"PPO_100k_dow_{trade_data_end_date}", 
                             timesteps=config.PPO_ts, previous_model=previous_models["PPO"], 
                             seed=seed, early_stopping=early_stopping)
        previous_models["PPO"] = model_ppo
        logging.info(f"======PPO Validation from {train_data_end_date} to {val_data_end_date}")
        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(trade_data_end_date)
        logging.info(f"PPO Sharpe Ratio: {sharpe_ppo}")

        logging.info("======DDPG Training========")
        model_ddpg = train_DDPG(env_train, f"DDPG_10k_dow_{trade_data_end_date}", 
                                timesteps=config.DDPG_ts, previous_model=previous_models["DDPG"], 
                                seed=seed, early_stopping=early_stopping)
        previous_models["DDPG"] = model_ddpg
        logging.info(f"======DDPG Validation from {train_data_end_date} to {val_data_end_date}")
        DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ddpg = get_validation_sharpe(trade_data_end_date)
        logging.info(f"DDPG Sharpe Ratio: {sharpe_ddpg}")

        logging.info("======TD3 Training========")
        logging.info(f"======TD3 Training timestep: {config.TD3_ts}")
        model_td3 = train_TD3(env_train, f"TD3_30k_dow_{trade_data_end_date}", 
                             timesteps=config.TD3_ts, previous_model=previous_models["TD3"], 
                             seed=seed, early_stopping=early_stopping)
        previous_models["TD3"] = model_td3
        logging.info(f"======TD3 Validation from {train_data_end_date} to {val_data_end_date}")
        DRL_validation(model=model_td3, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_td3 = get_validation_sharpe(trade_data_end_date)
        logging.info(f"TD3 Sharpe Ratio: {sharpe_td3}")

        logging.info("======SAC Training========")
        model_sac = train_SAC(env_train, f"SAC_30k_dow_{trade_data_end_date}", 
                             timesteps=config.SAC_ts, previous_model=previous_models["SAC"], 
                             seed=seed, early_stopping=early_stopping)
        previous_models["SAC"] = model_sac
        logging.info(f"======SAC Validation from {train_data_end_date} to {val_data_end_date}")
        DRL_validation(model=model_sac, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_sac = get_validation_sharpe(trade_data_end_date)
        logging.info(f"SAC Sharpe Ratio: {sharpe_sac}")

        a2c_sharpe_list.append(sharpe_a2c)
        ppo_sharpe_list.append(sharpe_ppo)
        ddpg_sharpe_list.append(sharpe_ddpg)
        td3_sharpe_list.append(sharpe_td3)
        sac_sharpe_list.append(sharpe_sac)

        # Select the model with highest Sharpe ratio
        max_model_sharpe = np.max([sharpe_a2c, sharpe_ddpg, sharpe_ppo, sharpe_td3, sharpe_sac])
        if max_model_sharpe == sharpe_ppo:
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif max_model_sharpe == sharpe_a2c:
            model_ensemble = model_a2c
            model_use.append('A2C')
        elif max_model_sharpe == sharpe_ddpg:
            model_ensemble = model_ddpg
            model_use.append('DDPG')
        elif max_model_sharpe == sharpe_td3:
            model_ensemble = model_td3
            model_use.append('TD3')
        else:
            model_ensemble = model_sac
            model_use.append('SAC')

        ############## Training and Validation ends ##############
        ############## Trading starts ##############
        logging.info(f"======Trading from {val_data_end_date} to {trade_data_end_date}")
        logging.info(f"Used Model: {model_use[-1]}")

        last_state_ensemble = DRL_prediction(df=df, 
                                           model=model_ensemble, 
                                           name="ensemble",
                                           last_state=last_state_ensemble, 
                                           iter_num=i,
                                           unique_trade_date=unique_trade_date,
                                           rebalance_window=rebalance_window,
                                           turbulence_threshold=turbulence_threshold,
                                           is_initial=is_initial,
                                           seed=seed)
        ############## Trading ends ##############

        # update
        seed_iter += 1

    train_end = time.time()
    logging.info(f"Training time: {(train_end - train_start) / 60} minutes")
    logging.info("============End Ensemble Strategy============")