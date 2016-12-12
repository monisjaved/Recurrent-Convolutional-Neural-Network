import numpy as np

__author__ = 'Moonis Javed'

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the data-set at each iteration.

    :type n: integer
    :param n: Size of the data set

    :type minibatch_size: int
    :param minibatch_size: size of mini-batch

    :type shuffle: bool
    :param shuffle: To shuffle the data set or not
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)
