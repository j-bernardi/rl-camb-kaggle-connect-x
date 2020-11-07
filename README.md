# RL Cambridge Kaggle Competition Project

This is a repository for members of the Cambridge (UK) Reinforcement Learning Meetup to create agents and compete in games of connect-X, an environment introduced by Kaggle. See for more details: https://www.kaggle.com/c/connectx

## Setup

### Environment

A conda environment is available which includes the Kaggle Connect-X RL environment and all modules that were available in the original competition:
- Python standard libraries (python 3.6)
- pytorch
- numpy
- scipy
- gym
- kaggle-environments==0.1.6 (more recent versions threw an error when importing `kaggle_environments`

To get the conda environment that allows you to use the kaggle environment, follow these steps. Check out README.md in `conda_envs` for more information on downloading `anaconda` before doing the below, if you don't have it already.

```bash
cd conda_env
conda create -f conda_env.yml
conda activate common
cd ..
```

### Getting started

After setting up the environment, try:
```bash
source setup_python_path.sh
python check_run.py
```

The `check_run.py` script checks that your environment is functioning with the environment correctly, and gives key examples on how to interact with the python code and get started. It is largely a copy-paste of the codeblocks available in the notebook: https://www.kaggle.com/ajeffries/connectx-getting-started

## Competing

Envisioning everyone makes their own agents and we compete!

TODO: create a file / env that will play agents against each other

TODO: create a leaderboard

