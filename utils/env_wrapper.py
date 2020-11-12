import gym
import numpy as np
import kaggle_environments as ke

class ConnectX(gym.Env):
    """A class to wrap the kaggle_environments connectx env

    Inspired by (read: mostly copied):
    https://www.kaggle.com/phunghieu/connectx-with-deep-q-learning
    """

    def __init__(
        self, score_target=80, out_of=100, debug=False,
        step_r=0.5, lose_r=0., draw_r=0.5, win_r=1.):
        # Set the env
        # score target = won of last out_of
        self.env = ke.make("connectx", debug=debug)
        self.configuration = self.env.configuration

        # (lose, draw, win) tuple
        self.standard_rewards = self.env.specification.reward.enum

        self.step_r = step_r
        self.lose_r = lose_r
        self.draw_r = draw_r
        self.win_r = win_r

        self.score_target = score_target
        self.out_of = out_of
        
        # Training config
        self.pair = [None, "random"]
        self.trainer = self.env.train(self.pair)

        # Define required gym fields (examples):
        self.action_space = gym.spaces.Discrete(
            self.configuration.columns)

        self.observation_space = gym.spaces.Discrete(
            self.configuration.columns * self.configuration.rows)

    def check_if_solved(self, game_scores):
        """Define the winning condition, for the agent.

        Figure out if you've won enough games out of the last
        self.out_of games to constitute .

        Returns:
            solved: whether the agent has (robustly) won
            score_dict: a collection of the last self.out_of
                games' outcomes
        """
        if len(game_scores) > self.out_of:

            score_vals, counts = np.unique(
                game_scores[-self.out_of:], return_counts=True
            )
            # Ensure game scores are consistent
            score_vals = np.round(score_vals.astype(float), 1)
            # Check we haven't messed up what's going in the score dict
            assert all(v in self.standard_rewards for v in score_vals),\
                f"{score_vals} != {self.standard_rewards}"
            score_dict = dict(zip(score_vals, counts))
            solved = score_dict.get(
                self.standard_rewards[2], 0) > self.score_target
        else:
            score_dict = None
            solved = False

        return solved, score_dict

    def custom_reward(self, reward, done):
        """
        Experiment ideas:
          - Go for the win! Penalise loss as much as draw
          - Take your time. Ignore negative steps
        """

        if reward == self.standard_rewards[2]:
            return self.win_r
        elif reward == self.standard_rewards[0]:
            return self.lose_r
        elif reward == self.standard_rewards[1] and done:
            return self.draw_r
        elif reward == self.standard_rewards[1] and not done:
            return self.step_r
        else:
            raise RuntimeError(
                f"Unexpected reward {reward} not in {self.standard_rewards}, "
                f"done: {done}"
        )

    def switch_trainer(self, switch_to):
        """
        Switch training pair to a new kaggle or 
        self-implemented agent. E.g.
            "negamax"
            "my_agent.py"
        """
        self.pair[1] = switch_to
        self.trainer = self.env.train(self.pair)

    def step_trainer(self, action):
        return self.trainer.step(action)
    
    def reset_trainer(self):
        return self.trainer.reset()
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
