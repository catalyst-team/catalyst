import random
import numpy as np


def set_global_seed(seed: int, deterministic: bool = None) -> None:
    """
    Sets random seed into PyTorch, TensorFlow, Numpy and Random.
    Additionally sets CuDNN to deterministic or non-deterministic mode

    Args:
        seed: random seed
        deterministic: deterministic mode if running in CuDNN backend.
            Setting it to ``True`` may slow down your training.
    """

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic is not None:
            # CuDNN reproducibility
            # https://pytorch.org/docs/stable/notes/randomness.html#cudnn
            torch.backends.cudnn.deterministic = deterministic
            # https://discuss.pytorch.org/t/how-should-i-disable-using-cudnn-in-my-code/38053/4
            torch.backends.cudnn.benchmark = not deterministic
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class Seeder:
    """
    A random seed generator.

    Given an initial seed,
    the seeder can be called continuously to sample a single
    or a batch of random seeds.

    .. note::

        The seeder creates an independent RandomState to generate random
        numbers. It does not affect the RandomState in ``np.random``.

    Example::
        >>> seeder = Seeder(init_seed=0)
        >>> seeder(size=5)
        [209652396, 398764591, 924231285, 1478610112, 441365315]

    """
    def __init__(self, init_seed: int = 0, max_seed: int = None):
        """
        Initialize the seeder.

        Args:
            init_seed (int, optional):
                Initial seed for generating random seeds. Default: ``0``.
        """
        assert isinstance(init_seed, int) and init_seed >= 0, \
            f"expected non-negative integer, got {init_seed}"

        self.rng = np.random.RandomState(seed=init_seed)

        # Upper bound for sampling new random seeds
        self.max = max_seed or np.iinfo(np.int32).max

    def __call__(self, size=1):
        """
        Return the sampled random seeds according to the given size.

        Args:
            size (int or list): The size of random seeds to sample.

        Returns
        -------
        seeds : list
            a list of sampled random seeds.
        """
        seeds = self.rng.randint(low=0, high=self.max, size=size).tolist()

        return seeds
