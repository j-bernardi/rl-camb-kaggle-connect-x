#!/usr/bin/env python3
import os
import argparse

import kaggle_environments as ke

try:
    from utils.utils import (
    	compare_agents, load_agent, run_agent_game
    )
except ModuleNotFoundError as mnfe:
    raise ModuleNotFoundError(
        f"{mnfe}\nDid you run source set_python_path.sh?"
)

def parse():
    parser = argparse.ArgumentParser(
        description="Loads two or one agents and runs "
                    "against the negamax agent or each other"
    )
    parser.add_argument(
        "agent_file1", 
        help="Must include encapsulating function 'my_agent(obs, config)'")
    
    parser.add_argument("agent_file2", default="negamax", nargs="?")

    parser.add_argument("games", default=10, nargs="?")

    return parser.parse_args()


def run_main(agent_file1, agent_file2, num_episodes):
    """Load one or two agents from filenames 

    Load one or two agents from filenames and run
    against each other. Agent file 2 can be a kaggle
    provided agent.

    """

    agent1 = load_agent(agent_file1)

    if os.path.exists(agent_file2):
        agent2 = load_agent(agent_file2)
    else:
        # assume it's a kaggle-compatible string
        agent2 = agent_file2
    
    print("Make env")
    env = ke.make("connectx", debug=True)
    env.render()
    
    env.reset()

    print("Run a single game (test, not recorded)")
    run_agent_game(env, agent1, "random", render=True)
    env.render()
    print(
        "\n***If there was an error, script won't fail here"
        "but agent is likely kaput.***\n"
    )
    
    print("Do evaluation")
    env.reset()
    mean_scores = compare_agents(
    	env, agent1, agent2, num_episodes=num_episodes)
    print("Complete")


if __name__ == "__main__":

    args = parse()
    run_main(args.agent_file1, args.agent_file2, args.games)
