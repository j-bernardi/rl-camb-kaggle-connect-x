import os
import numpy as np
import torch as tc
import torch.nn as nn

import kaggle_environments as ke


class FFDNN(nn.Module):
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


class DeepQLearner:
    """The trainer, to be serialised"""

    def __init__(self, env, discount=0.9, learning_rate=1.0):

        print(f"Creating agent with {env.configuration}")
        self.env = env
        self.square_options = 3  # empty, player 1, player2
        self.action_range = env.configuration.columns

        self.discount = discount
        self.learning_rate = learning_rate

        # E.g. for board states rows*cols that can be in position 0, 1, 2
        self.state_len = env.configuration.rows * env.configuration.columns

        # try init at high, 0 and low
        # Hypothesise starting high because need to learn when you get BEATEN
        self.model = FFDNN(self.state_len, self.action_range)
        self.loss_function = nn.MSELoss()
        self.optimizer = tc.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

    def do_training_episode(self, render=True):
        trainer = self.env.train([None, "random"])
        obs = trainer.reset()

        rewards = []
        while not env.done:
            state_tensor = tc.as_tensor(np.expand_dims(obs.board, 0), dtype=tc.float)
            action_predictions = self.model(state_tensor)

            # Remove values where the max value is a full column by min,
            # -1. so that if empty column has min value, it is still picked
            less_than_min_value = tc.min(action_predictions) - 1.
            non_full_columns = tc.where(
                tc.tensor(  # where col is not full
                    obs.board[:self.env.configuration["columns"]]) == 0,
                action_predictions,
                less_than_min_value  # avoid selection if col is full
            )

            if not env.done:
                action = tc.argmax(non_full_columns, dim=-1)
                next_obs, reward, done, info = trainer.step(int(action))
                rewards.append(reward)
                # if render:
                #     env.render(mode="ipython", width=100, height=90, header=False, controls=False)
                self.update(
                    tc.as_tensor(np.expand_dims(obs.board, 0), dtype=tc.float),
                    action, reward, 
                    tc.as_tensor(np.expand_dims(next_obs.board, 0), dtype=tc.float))
            obs = next_obs

        if render:
            env.render()  # final state

        return rewards

    def update(self, s, a, r, nxt_s):
        """Update the model with Bellman equation
        
        TODO: introduce if done, 0?
        """

        # Create the "label"
        q_prediction = tc.max(self.model(nxt_s), dim=-1)[0]  # max (1 is index)
        target_q = r + self.discount * q_prediction

        # Create the "predicted value" - at action a
        q_preds_from_state = self.model(s)
        gather_indices = tc.arange(a.shape[0]) * q_preds_from_state.shape[-1] + a
        q_preds_from_state_at_a = tc.gather(
            tc.reshape(q_preds_from_state, [-1]),
            -1,
            gather_indices
        )

        loss = tc.mean(self.loss_function(target_q, q_preds_from_state_at_a))
        loss.backward()
        self.optimizer.step()

        return loss  # todo - VALUE?

    def save(self, filename="model.pt"):

        print(f"Saving {filename} (whole model, includes file path)")
        tc.save(self.model, filename)

    def save_state_dict(self, filename="model_state_dict.pt"):
        print(f"Saving {filename} (state dict only)")
        tc.save(self.model.state_dict(), filename)

    def load(self, filename="model.pt"):

        print(f"Loading {filename}")
        self.model = tc.load(filename)

    def load_from_state_dict(self, filename="model_state_dict.pt"):
        print(f"Loading model from state dict {filename}")
        self.model = FFDNN(self.state_len, self.action_range)
        state_dict = tc.load(filename)
        self.model.load_state_dict(state_dict)
        self.model.eval()


if __name__ == "__main__":

    # View init
    env = ke.make("connectx")  # , {"rows": 4, "columns": 4}, debug=True)
    env.render()
    agent = DeepQLearner(env)

    model_name = "model10k.pt"
    state_dict_name = "model_state_dict10k.pt"
    
    # Train, player 1 against random
    if not os.path.exists(model_name):
        print("Train agent")
        for i in range(10000):
            if i % 100 == 0:
                print("Trained", i)
            agent.do_training_episode(render=False)
        agent.save(model_name)
        agent.save_state_dict(state_dict_name)
    else:
        # safer, doesn't require file reference
        agent.load_from_state_dict(state_dict_name)
