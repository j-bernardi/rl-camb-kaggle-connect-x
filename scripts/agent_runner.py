#!/usr/bin/env python3
import os
import argparse

import kaggle_environments as ke

from utils.utils import run_agent_game, load_agent


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
    # TODO: possibly enforce agent_function_name?
    parser.add_argument("agent_file1")
    parser.add_argument("agent_file2", default="negamax", nargs="?")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()
    run_main(args.agent_file1, args.agent_file2)