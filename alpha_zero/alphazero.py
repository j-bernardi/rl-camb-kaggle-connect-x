import os
import copy
import numpy as np
import torch as tc
import torch.nn as nn
import kaggle_environments as ke
from kaggle_environments.envs.connectx.connectx import is_win

from collections import deque

from utils.utils import compare_agents
from utils.env_wrapper import ConnectX


# TODO - they output a 19x19x1 for policy - prob of each square
#  So I want prob of each column...
# They do 40 residual layers followed by policy head, value head
class FFDNN(nn.Module):

    def __init__(self, insize, action_space):

        super(FFDNN, self).__init__()
        in_dims = 1
        for d in insize:
            in_dims *= d
        self.input = nn.Linear(in_dims, 64)
        self.value_layer1 = nn.Linear(64, 32)
        self.value_layer2 = nn.Linear(32, 1)

        self.action_layer1 = nn.Linear(64, 32)
        self.action_layer2 = nn.Linear(32, action_space)

    def forward(self, x):
        # print("x", x.shape)
        flattened_x = tc.flatten(x, start_dim=1)
        # print(flattened_x.shape)
        x = tc.tanh(
            self.input(flattened_x)
        )
        value_x = tc.tanh(self.value_layer1(x))
        action_x = tc.tanh(self.action_layer1(x))

        value_x = tc.tanh(self.value_layer2(value_x))
        action_x = tc.tanh(self.action_layer2(action_x))

        return value_x, action_x


class Node():

    def __init__(self, action_space, state_stack, parent=None):
        """Create a blank node object from current state"""
        self.n_children = action_space

        self.children = {i: None for i in range(self.n_children)}

        self.state_stack: StateStack2 = state_stack
        
        # num times each child action has been taken
        self.N = np.zeros((self.n_children,), dtype=int)
        
        # total value of each next state
        self.W = np.zeros((self.n_children,))
        
        # prior prob of selecting each action (from policy)
        # self.P = np.zeros((num_children,))

    # TODO determine what U(P, N) is
    def U(self, action_prior):
        return action_prior * self.N

    def Q(self):
        return np.where(self.N != 0, np.divide(self.W, self.N), 0.)


# TODO update stack on stepping, return done or not (as above)
class StateStack2():

    def __init__(self, num_rows=6, num_cols=7, inarow=4, player_stack_size=5, preload_state=None):

        # TODO should configure with num rows etc
        self.env = ke.make("connectx")
        self.config = self.env.configuration
        obs = self.env.reset()

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.inarow = inarow
        self.player_stack_size = player_stack_size

        stack_shape = (self.num_rows, self.num_cols, player_stack_size * 2 + 1)
        if preload_state is None:
            self.stack = np.zeros(
                stack_shape,
                dtype=np.uint8
            )
            self.first_empty_row = np.zeros((self.num_cols,), dtype=np.uint8)
        else:
            self.stack = preload_state
            assert preload_state.shape == stack_shape
            self.init_first_empty_row()

        print("LATEST ENV", self.env)
        import pprint
        pprint.pprint(self.env.specification.observation)
        pprint.pprint(self.env.specification.configuration)
        pprint.pprint(self.env.specification.action)

        print("EXAMPLE OBS")
        print(obs)
        print("Reshape")
        print(self.obs_to_grid(obs[0]["observation"]["board"]))
        assert obs[0]["observation"]["board"]\
            == obs[1]["observation"]["board"]

    def shared_step(self, player, action):
        if not np.all(self.stack[:, :, -1] == player):
            raise ValueError(
                f"Next to play {self.stack[:, :, -1]} != {player}")
        if self.env.state[player]["status"] == "INACTIVE":
            raise ValueError(
                f"Unexpected usage - must be for next player: {player}")

        print("Playing", player, "action", action)
        if player == 0:
            act = [action, None]
        elif player == 1:
            act = [None, action]
        else:
            raise ValueError(f"2 players only! {player}")

        return act

    # TODO two step functions are same except self. becomes hypothetical
    # Perhaps class function?

    # TODO could possibly speed up
    def init_first_empty_row(self):
        # Fill in first_empty_row
        for col in range(self.stack.shape[0]):
            row = state[col, :, 0] # + state[col, :, first_empty_row]
            empty_positions = [i for i, c in enumerate(row) if c == 0]

            if empty_positions:
                first_empty_row[col] = empty_positions[-1]
            else:
                # full
                first_empty_row[col] = state.shape[1]
        self.first_empty_row = first_empty_row

    def hypothetical_step(self, player, action):

        act_tuple = self.shared_step(player, action)

        hypothetical_env = ke.make("connectx")
        hypothetical_env.state = copy.deepcopy(self.env.state)

        new_obs = hypothetical_env.step(act_tuple)
        done = (
            hypothetical_env.state[player]["status"] == "DONE"
            or hypothetical_env.state[(player + 1) % 2]["status"] == "DONE"
        )
        # Just curious
        if hypothetical_env.state[player]["status"] == "DONE":
            assert hypothetical_env.state[
                (player + 1) % 2]["status"] == "DONE"

        print("NEW OBS hypothetical")
        new_obs_as_grid = self.obs_to_grid(new_obs[0]["observation"]["board"])
        print(new_obs_as_grid)

        assert (
            new_obs[0]["observation"]["board"]
            == new_obs[1]["observation"]["board"])

        # TODO - build the stack - only update whichever player just played

        hypothetical_stack = self.stack.copy()
        state_stack = hypothetical_stack[:, :,
            player * self.player_stack_size : 
            player * self.player_stack_size + self.player_stack_size]

        assert state_stack.shape == (
            hypothetical_stack.shape[:2]) + (self.player_stack_size,)

        # ROLL (by reassigning)
        state_stack[:, :, 1:self.player_stack_size] = state_stack[
            :, :, 0: self.player_stack_size - 1]
        # State stack only takes player number
        ones = np.uint8(1)
        zeros = np.uint8(0)
        state_stack[:, :, 0] = np.where(
            new_obs_as_grid == player + 1, ones, zeros)

        hypothetical_stack[:, :, -1] = (hypothetical_stack[0, 0:, -1] + 1) % 2

        return hypothetical_stack, done

    def step(self, player, action):
        act_tuple = self.shared_step(player, action)
        print("ACT TUPLE", act_tuple)
        # TODO - this needs to be hypothetical
        # Make an env and replace the gameboard arrays with current...
        new_obs = self.env.step(act_tuple)

        done = (
            self.env.state[player]["status"] == "DONE"
            or self.env.state[(player + 1) % 2]["status"] == "DONE"
        )
        # Just curious
        if self.env.state[player]["status"] == "DONE":
            assert self.env.state[(player + 1) % 2]["status"] == "DONE"

        print("NEW OBS, done:", done)
        new_obs_as_grid = self.obs_to_grid(new_obs[0]["observation"]["board"])
        print(new_obs_as_grid)
        assert (
            new_obs[0]["observation"]["board"]
            == new_obs[1]["observation"]["board"])

        # TODO - build the stack - only update whichever player just played

        state_stack = self.stack[:, :,
            player * self.player_stack_size : 
            player * self.player_stack_size + self.player_stack_size]

        assert state_stack.shape == (self.stack.shape[:2]) + (self.player_stack_size,)

        # ROLL (by reassigning)
        state_stack[:, :, 1:self.player_stack_size] = state_stack[
            :, :, 0: self.player_stack_size - 1]
        # State stack only takes player number
        ones = np.uint8(1)
        zeros = np.uint8(0)
        state_stack[:, :, 0] = np.where(
            new_obs_as_grid == player + 1, ones, zeros)

        self.stack[:, :, -1] = (self.stack[0, 0:, -1] + 1) % 2
        self.first_empty_row[action] += 1

        return done

    def obs_to_grid(self, arr):
        return np.reshape(
            arr,
            (self.config.rows, self.config.columns),
        )

    def grid_to_obs(self, arr):
        x = np.reshape(arr, (-1,))
        assert np.all(self.obs_to_grid(x) == arr)
        return x

    def __repr__(self):
        ret = "Game stack obj <StateStack2>"
        for i in reversed(range(self.player_stack_size)):
            ret += f"\nRound -{i}\n"
            grid = (
                self.stack[:, :, i]
                + 2 * self.stack[:, :, i + self.player_stack_size]
            )
            ret += f"{grid}"
            ret += f"\nWith player {(self.stack[0, 0, -1] + 1) % 2} to play"
        return ret

    def get_current_state(self):
        return (
                self.stack[:, :, 0]
                + 2 * self.stack[:, :, self.player_stack_size]
            )


class AlphaGo():

    def __init__(self, env):

        print(f"Creating agent with {env.configuration}")
        self.connectx = env
        self.config = env.configuration
        # E.g. for board states rows*cols that can be in position 0, 1
        self.state_shape = (self.config.rows, self.config.columns)
        self.action_space = env.configuration.columns

        self.self_play_iterations = 2  # 100
        self.evaluations = 50
        self.batch_size = 128
        self.memory_size = 5000
        self.memory = deque(maxlen=self.memory_size)

        self.move_history = 4
        self.discount = 1.
        self.learning_rate = 0.01
        self.num_simulations = 10  # ?
        self.inverse_tau = 1. / 10.  # TODO - example value?

        depth = (self.move_history + 1) * 2 + 1
        self.model = FFDNN(self.state_shape + (depth,), self.action_space)
        init_state = StateStack2(*self.state_shape)
        self.tree = Node(self.action_space, init_state)
        self.value_loss_f = nn.MSELoss()
        self.prob_loss_f = self.cross_entropy_from_logits  # nn.CrossEntropyLoss()
        self.optimizer = tc.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def cross_entropy_from_logits(self, pi, p):

        return tc.sum(tc.mul(self.softmax(pi), self.log_softmax(p)))

    def obs_to_grid(self, obs):
        return np.reshape(
            obs,
            (self.config.rows, self.config.columns)
        )

    def act(self, training=True):
        """
            state_stack (StateStack2):

        """
        player = 0  # TODO changeable
        def mask_invalid(first_empty, val, stochastic=False):
            allowed_action_idc = np.reshape(
                np.argwhere((first_empty < self.config.rows)),
                -1
            )
            print("Allowed idcs", allowed_action_idc)
            if not np.any(allowed_action_idc):
                print("RETURN NONE")
                return None
            print("ONE IS ALLOWED")

            allowed_vals = np.squeeze(val[allowed_action_idc])
            print("Allowed actions", allowed_vals)
            a_map = {i: a for i, a in enumerate(allowed_action_idc)}
            if stochastic:
                # choose from distribution
                valid_action =int(
                    np.random.choice(
                        range(len(allowed_vals)), 1, p=val/np.sum(val)))
            else:
                # choose argmax
                valid_action = np.argmax(allowed_vals)
            print("Action", valid_action, "->", a_map[valid_action])
            return a_map[valid_action]

        # TODO - state_obj needs to implement "finished"

        # Do simulations from the current state
        for s in range(self.num_simulations):
            # Root node is the current game state
            # For now make a fresh tree
            simulation_head = self.tree
            """
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
            Need to figure out what "keep the subtree for
            further calculations" means. Essentially I am setting
            the self.tree to a leaf, and should be going back to the root
            node for each simulation.
            >>>>>>>>>>>>>>>>>>>>>>>>>>>
            """
            state_obj = simulation_head.state_stack
            it = 0
            finished = False
            while not finished:
                print("it", s, it); it += 1
                with tc.no_grad():
                    value, action_dist = self.model(
                        tc.reshape(
                            tc.tensor(
                                state_obj.stack, dtype=tc.float32,
                                requires_grad=False),
                            (1,) + state_obj.stack.shape)
                    )
                # TODO mask out children that are none
                # print("Actions", action_dist, "vals", value)
                q = simulation_head.Q()
                u = simulation_head.U(action_dist.numpy()[0])
                # print("Q", q, "U", u)
                # print("First empty", simulation_head.state_stack.first_empty_row)
                # print("Where", simulation_head.state_stack.first_empty_row < self.config.rows)


                print("ORIG", np.argmax(q + u))
                # TODO - verify we're correctly masking invalid here!
                print("First empty row, outside", simulation_head.state_stack.first_empty_row)
                action_i = mask_invalid(
                    simulation_head.state_stack.first_empty_row, q + u)
                print("MASKED", action_i)
                if action_i is None:
                    # full grid - do next sim
                    break

                # Update choice of action
                print("CURRENT N", simulation_head.N)
                simulation_head.N[action_i] += 1
                print("AFTER ACTION", action_i, simulation_head.N)
                # TODO verify
                simulation_head.W += value.numpy()[0]

                next_state, finished = state_obj.hypothetical_step(
                    player=player, action=action_i)
                print("NEXT", next_state)
                print("FINISHED", finished)
                # state_obj.is_finished()

                # Create the child if this is the first time and not full col
                if (
                        simulation_head.children[action_i] is None
                        and simulation_head.state_stack.first_empty_row[
                            action_i] < self.config.rows
                        and not finished):

                    # TODO should copy num_rows
                    child_state = StateStack2(preload_state=next_state)
                    print("CREATING CHILD STARTING FROM", child_state)
                    print("CHILD\n", child_state)

                    simulation_head.children[action_i] = Node(
                        self.action_space, child_state)

                simulation_head = simulation_head.children[action_i]
                if simulation_head is None:
                    assert finished  # testing
                if it == 2:
                    break

        # Choose action TODO need to mask invalid actions
        if training:
            print("Acting on", self.tree.state_stack.stack)
            # print("Tree probs", self.tree.N)
            cooled_probs = np.where(
                self.tree.N > 0, self.tree.N ** self.inverse_tau, 0.)
            print("COOLED", cooled_probs)
            action = mask_invalid(
                self.tree.state_stack.first_empty_row, cooled_probs,
                stochastic=True)
            sample = cooled_probs
        else:
            action = int(
                mask_invalid(
                    self.tree.state_stack.first_empty_row, self.tree.N))
            sample = self.tree.N

        # Keep subtree for subsequent moves
        # TODO - verify - is this what's meant by keep rest of tree for
        # further calcs? It's None when we get to a leaf, btw
        self.tree = self.tree.children[int(action)] or Node(
            self.action_space, StateStack2(*self.state_shape)
        )

        return action, sample

    def do_training_episodes(self, train_against, render=True):

        # Trainer keeps the state of the EPISODE against adversary
        # TODO - one day train_against will have to be self!
        player = 0
        trainer = self.connectx.env.train([None, train_against])

        for game_idx in range(self.self_play_iterations):
            print("Game", game_idx)
            obs = trainer.reset()
            # game_rewards = []

            # State of the EPISODE as numpy array
            state = self.obs_to_grid(tc.tensor(obs.board))

            # Create a state stack object for this game - stores history for
            # actions
            game_state_stack = StateStack2(
                num_rows=self.config.rows, 
                num_cols=self.config.columns
                # TODO add inarow (from config)
            )

            done = False
            stp = 0
            while not done:
                current_state = game_state_stack.stack
                try:
                    action, search_probs = self.act(game_state_stack)
                except TypeError as te:
                    env.render()
                    print("Not updated^")
                    raise te

                # Next obs, info not needed
                next_obs, reward, done, _ = trainer.step(int(action))

                # game_rewards.append(reward)
                self.memory.append(
                    (current_state, search_probs, reward))

                if reward is None:
                    # DIAGNOSE
                    print("Rendering")
                    env.render()
                    print("action taken", action)
                    print("Done?", done)
                    sys.exit()

                # Update the stack to keep the history
                game_state_stack.step(player=player, action=action)

                # TODO current_state is not updating after step
                print("CURRENT STATE", game_idx, stp)
                print(game_state_stack.get_current_state())
                print("ENV STATE")
                new_grid = self.obs_to_grid(next_obs["board"])
                print(new_grid)

                assert np.all(game_state_stack.get_current_state() == next_obs)
                stp += 1

            if render:
                print("Game", game_idx)
                env.render()  # final state

        self.train_network()

        # return rewards

    def train_network(self):
        """Update the model"""

        # Get training batch
        minibatch_is = np.random.choice(
            len(self.memory),
            size=min(self.batch_size, len(self.memory)),
            replace=False,
            p=None
        )
        minibatch = [self.memory[i] for i in minibatch_is]
        to_float_tensor = lambda x: tc.tensor(x, dtype=tc.float32)
        s, pi, r = tuple(map(to_float_tensor, zip(*minibatch)))

        current_model_v, current_model_p = self.model(s)
        current_model_v = tc.squeeze(current_model_v)
        current_model_p = tc.squeeze(current_model_p)
        # assert current_model_v.shape

        loss = (
            self.value_loss_f(r, current_model_v)
            + self.prob_loss_f(pi, current_model_p)
            # + regularisation
        )
        loss.backward()
        self.optimizer.step()

        return loss  # todo - VALUE?

    def evaluate(self, against_file):

        # TODO with kaggle env
        mean_scores = compare_agents(
            self.submission_path, against_file,
            num_episodes=self.evaluations
        )

        wins = sum([1 for g in mean_scores if g == 1.])
        return wins / self.evaluations >= 0.55

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
    env = ConnectX()  # , {"rows": 4, "columns": 4}, debug=True)
    env.render()
    agent = AlphaGo(env)

    model_name = "alphazero.pt"
    state_dict_name = "model_state_dict_alphazero.pt"

    num_training_loops = 2000
    evaluate_after = 100
    current_best_agent = "random"  # AlphaGo(env)  # random

    # Train, player 1 against random
    if not os.path.exists(model_name):

        print("Train agent")

        for i in range(num_training_loops):
            if i % 100 == 0:
                print("Trained", i)

            agent.do_training_episodes(
                current_best_agent, render=True)

            if i % evaluate_after == 100:
                agent_is_better = agent.evaluate(current_best_agent)
                if agent_is_better:
                    # TODO update current best to this agent
                    pass

        agent.save(model_name)
        agent.save_state_dict(state_dict_name)

    else:
        # safer, doesn't require file reference
        agent.load_from_state_dict(state_dict_name)

