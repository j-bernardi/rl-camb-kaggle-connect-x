#!/usr/bin/env python3
import os
import argparse

import kaggle_environments as ke

try:
    from utils.utils import run_agent_game, load_agent
except ModuleNotFoundError as mnfe:
    raise ModuleNotFoundError(
        f"{mnfe}\nDid you run source set_python_path.sh?"
)


def run_main(agent_file1, agent_file2):
    """Load one or two agents from filenames 

    Load one or two agents from filenames and run
    against each other or the negamax agent, respectively

    """

    agent1 = load_agent(agent_file1)

    if os.path.exists(agent_file2):
    	agent2 = load_agent(agent_file2)
    else:
    	# assume it's a kaggle-compatible string
    	agent2 = agent_file2

    env = ke.make("connectx")
    env.reset()
    run_agent_game(env, agent1, agent2)


def parse():
    parser = argparse.ArgumentParser(
        description="Loads two or one agents and runs "
                    "against the negamax agent or each other"
    )
    parser.add_argument(
        "agent_file1", 
        help="Must include encapsulating function 'my_agent(obs, config)'")
    parser.add_argument("agent_file2", default="negamax", nargs="?")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()
    run_main(args.agent_file1, args.agent_file2)
