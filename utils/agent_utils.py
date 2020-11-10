import numpy as np
import torch as tc


def obs_to_tensor(obs):
	return tc.as_tensor(obs.board, dtype=tc.float)


def get_batch_from_memory(memory, batch_size):

    minibatch_i = np.random.choice(
        len(memory),
        min(batch_size, len(memory)),
    )
        
    minibatch = [memory[i] for i in minibatch_i]
    
    def as_tensor(xs):
        try:  # temo debug for over-complete episodes
            if tc.is_tensor(xs[0]):
                return tc.stack(xs)
            else:
                return tc.as_tensor(xs)
        except RuntimeError as re:
            print("XS", xs)
            raise re

    args_as_tuple = tuple(map(as_tensor, zip(*minibatch)))

    return args_as_tuple
