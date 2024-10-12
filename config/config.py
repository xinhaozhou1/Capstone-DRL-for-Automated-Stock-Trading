# Used to store the configuration of the project
import datetime
import os

# data
# Training data settings: Dow Jones 30 (DJ30)
dj_start_date = '2009-01-01'
dj_end_date = '2020-12-31'
dj_stock_ticker = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX',
                   'DD', 'DIS', 'GS', 'HD', 'IBM', 'INTC',
                   'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
                   'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV',
                   'UNH', 'V', 'VZ', 'WBA', 'WMT', 'XOM']

# Env parameters
init_turbulence_sample_start_date = 20090000
init_turbulence_sample_end_date = 20151000
trade_start_date = 20151001
trade_end_date = 20200707
rebalance_window = 63
validation_window = 63

# Directories
config_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(config_dir)
dj_data_path = os.path.join(project_root, "data", f"dow_30_{dj_start_date}_{dj_end_date}.csv")

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
trained_model_dir = f"trained_models/{now}"
trained_model_dir = os.path.join(project_root, trained_model_dir)
os.makedirs(trained_model_dir)

results_dir = f"results/{now}"
results_dir = os.path.join(project_root, results_dir)
os.makedirs(results_dir)