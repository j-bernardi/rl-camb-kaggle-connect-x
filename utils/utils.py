import sys
import inspect

import kaggle_environments as ke


def run_agent_game(env, agent1, agent2, render=True):
    """Run two agents (custom or standard) on an env
    
    NOTE: env.reset() required if running fresh

    TODO: args

    """

    env.run([agent1, agent2])
    if render:
        env.render(mode="ipython", width=500, height=450)  # TODO run in ipython or another render


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
