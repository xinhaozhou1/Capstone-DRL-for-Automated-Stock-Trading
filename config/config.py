# Used to store the configuration of the project
import datetime
import os

# data
# Training data settings: Dow Jones 30 (DJ30)
config_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(config_dir)

dj_start_date = '2009-01-01'
dj_end_date = '2020-12-31'
dj_stock_ticker = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX',
                   'DD', 'DIS', 'GS', 'HD', 'IBM', 'INTC',
                   'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
                   'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV',
                   'UNH', 'V', 'VZ', 'WBA', 'WMT', 'XOM']

dj_data_path = os.path.join(project_root, "data", f"dow_30_{dj_start_date}_{dj_end_date}.csv")