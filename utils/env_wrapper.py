import gym
import kaggle_environments as ke

class ConnectX(gym.Env):
    """A class to wrap the kaggle_environments connectx env

    Inspired by (read: mostly copied):
    https://www.kaggle.com/phunghieu/connectx-with-deep-q-learning
    """

    def __init__(self, debug=False):
        # Set the env
        self.env = ke.make("connectx", debug=debug)
        self.configuration = self.env.configuration
        
        # Training config
        self.pair = [None, "random"]
        self.trainer = self.env.train(self.pair)

        # Define required gym fields (examples):
        self.action_space = gym.spaces.Discrete(
            self.configuration.columns)
        self.observation_space = gym.spaces.Discrete(
            self.configuration.columns * self.configuration.rows)

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
