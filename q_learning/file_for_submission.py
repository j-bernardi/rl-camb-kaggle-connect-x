"""
Manually written file which I think should be compatible 
with load_agent. It's ugly, but it's not as ugly as
writing all the model weights in as variables.

https://www.kaggle.com/ajeffries/connectx-getting-started

TODO: Verify this actually works in the competition.
    I really hope this avoids writing model weights in,
    but there's a chance it doesn't and there's no 
    avoiding it. Could possibly turn model_state_dict.pt
    into json and dump it as a variable here. Sigh.
    E.g. https://www.kaggle.com/phunghieu/connectx-with-deep-q-learning

TODO: make the write_agent script that could have made
this automatically (tall order!)
   The submit_agent.py script probably can't handle
   this, thus you must make this file on your own 
"""


def my_agent(observation, configuration):
    import torch as tc
    import torch.nn as nn
    import numpy as np
    import kaggle_environments as ke


    class FFDNN(nn.Module):
        """
        MUST match the original configuration of whatever model
        the state_dict below was saved from.

        It also must load in 5 seconds (I think, but possibly 2), 
        as the game has a timeout.
        """
        def __init__(self, insize, action_space):

            super(FFDNN, self).__init__()
            self.input = nn.Linear(insize, 64)
            self.layer1 = nn.Linear(64, 32)
            self.layer2 = nn.Linear(32, action_space)

        def forward(self, x):
            x = tc.tanh(self.input(x))
            x = tc.tanh(self.layer1(x))
            x = self.layer2(x)
            return x


    def get_model_from_state_dict(path="model_state_dict10k.pt"):
        """
        Again, the FFDNN code above must match that defined in 
        whatever model saved model_state_dict.pt. It's ugly,
        but so is the kaggle submission format (IMO)!

        Haven't verified that model_state_dict.pt would actually
        be available to 
        """

        example_env = ke.make("connectx")  # , {"rows": 4, "columns": 4})
        # model = tc.load("model.pt")
        model = FFDNN(
            example_env.configuration["rows"] 
                * example_env.configuration["columns"], 
            example_env.configuration["columns"]
        )
        model.load_state_dict(tc.load(path))
        model.eval()
        return model

    model = get_model_from_state_dict()  # loads every time

    # Use the model to make a prediction
    state_tensor = tc.as_tensor(
        np.expand_dims(observation.board, 0),
        dtype=tc.float
    )
    action_predictions = model(state_tensor)

    # Remove values where the max value is a full column by min,
    # -1. so that if empty column has min value, it is still picked
    less_than_min_value = tc.min(action_predictions) - 1.
    non_full_columns = tc.where(
        tc.tensor(  # where col is not full
            observation.board[:configuration["columns"]]) == 0,
        action_predictions,
        less_than_min_value  # avoid selection if col is full
    )
    return int(tc.argmax(non_full_columns))
