# RL Cambridge Kaggle Competition Project

This is a repository for members of the Cambridge (UK) Reinforcement Learning Meetup to create agents and compete in games of connect-X, an environment introduced by Kaggle. See for more details: https://www.kaggle.com/c/connectx

## Setup

### Environment

A conda environment is available which includes: the environment, and all modules that were available in the original competition. That is:
- Python standard libraries (py3.6)
- pytorch
- numpy
- scipy

OpenAI's gym module was originally allowed, but is currently omitted from the env until required. Check out README.md in `conda_envs` for more information on downloading anaconda, if you don't have it already.

To get the conda environment that allows you to use the kaggle environment:

`cd conda_env`
`conda create -f conda_env.yml`
`conda activate common`

### Getting started

The `check_run.py` script checks that your environment is functioning with the environment correctly, and gives key examples on how to interact with the python code and get started.

It is a copy-paste of the codeblocks available in the notebook:

https://www.kaggle.com/ajeffries/connectx-getting-started

## The competition

Envisioning everyone makes their own agents and we compete!

TODO: create a file / env that will play agents against each other

TODO: create a leaderboard

