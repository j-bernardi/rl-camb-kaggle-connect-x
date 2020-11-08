import os
import sys
import inspect

import kaggle_environments as ke

import scripts.submit_agent as submit_agent


# To create the submission, an agent function should be 
# fully encapsulated (no external dependencies).
def my_agent(observation, configuration):
    """Encapsulated agent function
    
    Kaggle use the conda_env/conda_env.yml to run this agent. 
    Can only import:
      - pytorch (torch)
      - numpy
      - scipy
      - gym
      - Python 3.6 standard library

    Args:
        observation: the gameboard observation extracted with a 
            kaggle_environments connect-x env at env.observation.
            Format dict: {
                board: [
                    list len rowXcol 
                    with 0 (empty), 
                    1, 2 (player counters)], 
                mark: turn_marker}, where N is set by configuration.

        configuration: the dict stored in a kaggle_environments 
            env object: env.configuration. Keys (defaults): 
                'timeout': 5,
                'columns': 7,
                'rows': 6,
                'inarow': 4,  # A.k.a "X"
                'steps': 1000,

    Returns:
        The agent's choice of move as a column-integer,
        given the args
    """
    ## MAY CHANGE
    import random
    
    total_cols = configuration.columns
    non_full_columns = [
        c for c in range(total_cols) if observation.board[c] == 0]
    column_choice = random.choice(non_full_columns)
    ############
    return column_choice


def do_testing(env, agent1, agent2="random", render=True):
    env.reset()
    env.run([agent1, agent2])
    if render:
        env.render(mode="ansi", width=500, height=450)  # TODO run in ipython or another render


def do_training_episode(agent, trainer, render=True):
    obs = trainer.reset()
    rewards = []
    while not env.done:
        my_action = agent(obs, env.configuration)
        obs, reward, done, info = trainer.step(my_action)
        rewards.append(reward)
        if render:
            env.render(mode="ansi", width=100, height=90, header=False, controls=False)

        ### TRAINING STEPS GO HERE ###
    env.render()  # final state
    return rewards


if __name__ == "__main__":

    # View init
    env = ke.make("connectx", debug=True)
    env.render()
    
    # Train, player 1 against random
    print("Train agent")
    trainer = env.train([None, "random"])
    do_training_episode(my_agent, trainer)
    
    print("Run trained agent")
    # Run trained agent against random
    env.run([my_agent, "random"])
    env.render(mode="ansi", width=500, height=450)

    # Interactive play against an agent
    # Insert your agent as player 2
    print("Play against your agent")
    env.play([None, my_agent], width=500, height=450)

    # Write the agent to file
    submit_agent.run_main(__file__, "example_submission.py", force=True)
