import os
import sys
import copy
import random
import collections
import numpy as np
import importlib.util
import functools
import weakref


def merge_dicts(*dicts):
    """
    Recursive dict merge.
    Instead of updating only top-level keys,
        dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys.
    """
    assert len(dicts) > 1

    dict_ = copy.deepcopy(dicts[0])

    for merge_dict in dicts[1:]:
        for k, v in merge_dict.items():
            if (k in dict_ and isinstance(dict_[k], dict)
                    and isinstance(merge_dict[k], collections.Mapping)):
                dict_[k] = merge_dicts(dict_[k], merge_dict[k])
            else:
                dict_[k] = merge_dict[k]

    return dict_


def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i)
        torch.cuda.manual_seed_all(i)
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    random.seed(i)
    np.random.seed(i)


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def memorized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator


def str2params(string, delimeter="-"):
    try:
        result = list(map(int, string.split(delimeter)))
    except Exception:
        result = None
    return result


def create_if_need(path):
    if not os.path.exists(path):
        os.makedirs(path)


def boolean_flag(parser, name, default=False, help=None):
    """
    Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace("-", "_")
    parser.add_argument(
        "--" + name, action="store_true",
        default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def query_yes_no(question, default="no"):
    """
    Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {
        "yes": True, "y": True, "ye": True,
        "no": False, "n": False
    }
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: \"%s\"" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with \"yes\" or \"no\" "
                             "(or \"y\" or \"n\").\n")


class stream_tee(object):
    # Based on https://gist.github.com/327585 by Anand Kunal
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
        self.__missing_method_name = None  # Hack!

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        self.__missing_method_name = name  # Could also be a property
        return getattr(self, "__methodmissing__")

    def __methodmissing__(self, *args, **kwargs):
            # Emit method call to the log copy
            callable2 = getattr(self.stream2, self.__missing_method_name)
            callable2(*args, **kwargs)

            # Emit method call to stdout (stream 1)
            callable1 = getattr(self.stream1, self.__missing_method_name)
            return callable1(*args, **kwargs)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class FrozenClass(object):
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


class FakeSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass
