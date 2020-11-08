import sys
import inspect
import numpy as np

import kaggle_environments as ke


def run_agent_game(env, agent1, agent2, render=True):
    """Run two agents (custom or standard) on an env
    
    NOTE: env.reset() required if running fresh

    TODO: args

    """
    print("Running")
    env.run([agent1, agent2])
    if render:
        print("Rendering")
        env.render(mode="ansi", width=500, height=450)  # TODO run in ipython or another render


def compare_agents(env, agent1, agent2, num_episodes=10):
    rewards = ke.evaluate(
        "connectx", [agent1, agent2], num_episodes=num_episodes)
    try:
        mean_rewards = np.mean(rewards, axis=0)
    except TypeError as te:
        raise TypeError(
            f"{te}None-reward likely means your submission"
            f" file isn't runnable i.e. has an error in it."
        )
    print(
        "mean reward of agent1 vs agent2:",
        mean_rewards[0], ":", mean_rewards[1]
    )
    return mean_rewards


def write_agent_to_file(function, outfile):
    """Overwrites a file with a submitted access function

    TODO: args

    """
    source_code = inspect.getsource(function)
    with open(outfile, "w") as f:
        f.write(source_code)
    print(function, "written to", outfile)


def load_agent(agent_file: str):
    """Read a loaded, runnable agent
    
    Args:
        agent_file: A Kaggle-readable file name 
            (e.g. one saved with write_agent_to_file)

    Returns:
        TODO what is the type?
    """
    out = sys.stdout  # apparently needed as a "workaround"
    submission = ke.utils.read_file(agent_file)
    loaded_agent = ke.utils.get_last_callable(submission)
    sys.stdout = out
    return loaded_agent
