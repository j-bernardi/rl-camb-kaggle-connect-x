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

We can use this environment to compete between agents! If you have an agent that is in submittable format, we can compete. Submittable format means to have an entirely encapsulated function called `my_agent(observation, configuration)`. This is verified with `scripts/agent_runner my_agent_file.py`, or by jumping straight in and trying the below:

```bash
scripts/evaluate_agents.py agent1_file.py [agent2_file.py negamax random]
```

`example_submission/` shows the example agent and the example code used to create the submittable file (reproduced in `scripts/submit_agent.py`), as-provided by Kaggle.

`q_learning/` has an example written by the repo author that is runnable with the `evaluate_agents.py` script, but I have not yet verified that it will be runnable in a proper kaggle submission (the main uncertainty is whether the model state dict .pt file is submittable, as I have seen some examples where the state dict is written out as a variable within the function. I hope this won't be necessary, as it seems like a big pain!

# Leaderboard
Negamax (Kaggle) vs `q_learning` (j-bernardi): 0.0 : 1.0

^ Should be easy to beat me!
