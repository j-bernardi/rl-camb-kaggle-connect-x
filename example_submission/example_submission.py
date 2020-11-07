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
