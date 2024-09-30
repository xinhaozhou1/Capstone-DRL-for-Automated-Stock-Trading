# Capstone-DRL-for-Automated-Stock-Trading

## Installation
```shell
git clone https://github.com/xinhaozhou1/Capstone-DRL-for-Automated-Stock-Trading.git
```

### Prerequisites
We recommend creating a virtual environment and installing the required packages.

In anaconda prompt window, run the following commands. We recommend using Python 3.7.16.
```shel
conda create -n capstone python=3.7
conda activate capstone
```

Then cd into this repo, and install the dependencies in the requirements.txt.
*Pay attention, this `requirements.txt` is different from that in the paper repo.*
```shell
cd .../your/directory/Capstone-DRL-for-Automated-Stock-Trading
pip install -r requirements.txt
```

Because we need to use GAIL, DDPG, TRPO, and PPO1 parallelize training, we need to install OpenMPI besides the `stable-baselines[mpi] ` package installed.

Install [OpenMPI for Windows](https://www.microsoft.com/en-us/download/details.aspx?id=57467) (you need to download and install `msmpisetup.exe`).
For more information, the documentation for stable baselines is [here](https://stable-baselines.readthedocs.io/en/master/guide/install.html).

*The `stable-baselines[mpi]` package is deprecated, and has migrated into `stable-baselines3`. We will update the code to use `stable-baselines3` in the future.*

### Trouble Shooting
You may encounter the following error when running the code:

#### 1. TypeError: Descriptors cannot not be created directly

For the two options the machine provided, the first approach is recommended, which is to downgrade the `protobuf` package to 3.20.
*Make sure run this command in administrator mode in anaconda prompt window.*
```shell
pip install protobuf==3.20.* --user
```

#### 2. Python WindowsError: [Error 123] The filename, directory name, or volume label syntax is incorrect
Modify the source code a little bit in `config.py`
```python
## To avoid syntax error from os
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TRAINED_MODEL_DIR = f"trained_models/{now}"
os.makedirs(TRAINED_MODEL_DIR)
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
```

#### 3. Any further questions
Contact Xinhao Zhou for help!
