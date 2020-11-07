#!/usr/bin/env python3
import os
import sys
import inspect
import importlib
import argparse

import kaggle_environments as ke

from utils.utils import (
    run_agent_game, write_agent_to_file, load_agent
)


def run_main(agent_file, outfile, force=False):
    """Save an agent to a runnable file

    To create the submission, an agent function should be 
    fully encapsulated (no external dependencies).

    Args: 

        agent_file :file(.py) containing the agent. Must include
            an entirely self-contained function called my_agent,
            which takes env.observation and env.configuration as
            its two arguments and returns a move

        output: output(.py) file to save the agent 
            submission to. This is loadable by the kaggle utils
        
        force: Whether to force-overwrite the output file if it
            already exists"

    """

    agent_function = extract_function_code(agent_file)

    if os.path.exists(outfile) and not force:
        raise FileExistsError(
            f"File {outfile} exists, force not flagged")

    env = ke.make("connectx", debug=True)

    # Check the agent is valid and runs as expected
    print(f"Verifying agent {agent_function.__name__}... ", end="")
    env.reset()
    run_agent_game(env, agent_function, agent2="random", render=False)
    print("Agent verified.")

    print(
        f"Writing agent function my_agent from "
        f"{agent_file} to file {outfile}"
    )
    write_agent_to_file(agent_function, outfile)

    # Now check the written file is valid

    # Validate the submission
    print(f"Loading saved agent {outfile} to validate subission... ", end="")
    loaded_agent = load_agent(outfile)
    print("Loaded.")
    
    print("Verifying agent with play")
    env.reset()
    run_agent_game(env, loaded_agent, agent2="random")
    
    if env.state[0].status == env.state[1].status == "DONE":
        print("SUCCESS - status DONE")
    else:
        print("FAILED")
        print(f"{env.state[0].status}\n{env.state[1].status}")


def extract_function_code(agent_file: str):
    """Imports the agent function my_agent from the given file

    """

    agent_file_parent_folder = os.sep.join(
        agent_file.split(os.sep)[:-1]  # path/to
    )
    sys.path.append(agent_file_parent_folder)
    
    agent_module = importlib.import_module(
        agent_file.strip(".py").split(os.sep)[-1]
    )
    agent_function = getattr(agent_module, "my_agent")
    return agent_function


def parser():
    parser = argparse.ArgumentParser(
        description="Save your agent to a runnable file."
    )

    parser.add_argument("agent_file",
        help="file(.py) containing the agent"
    )

    # Currently opting to enforce this to be "my_agent"
    # parser.add_argument("function_name",
    #     help="Name of the function granting the access-point "
    #          "for the agent, that is, it takes only 2 arguments"
    #          " of observation, configuration, and returns your "
    #          "agent's action."
    # )

    parser.add_argument("output_file", 
        help="Output(.py) file to save the agent submission to. "
             "This is runnable against other agents."
    )

    parser.add_argument("-f", "--force", action="store_true",
        help="Whether to force-overwrite the output file if it"
             "already exists"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    run_main(args.agent_file, args.output_file, args.force)
