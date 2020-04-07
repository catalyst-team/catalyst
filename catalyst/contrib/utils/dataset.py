from typing import Callable, Dict, Tuple
from collections import defaultdict
import glob
import itertools
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DictDataset = Dict[str, object]


def create_dataset(
    dirs: str,
    extension: str = None,
    process_fn: Callable[[str], object] = None,
    recursive: bool = False,
) -> DictDataset:
    """
    Create dataset (dict like `{key: [values]}`) from vctk-like dataset::

        dataset/
            cat/
                *.ext
            dog/
                *.ext

    Args:
        dirs (str): path to dirs, for example /home/user/data/**
        extension (str): data extension you are looking for
        process_fn (Callable[[str], object]): function(path_to_file) -> object
            process function for found files, by default
        recursive (bool): enables recursive globbing

    Returns:
        dict: dataset
    """
    extension = extension or "*"
    dataset = defaultdict(list)

    dirs = [os.path.expanduser(k) for k in dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]

    for d in sorted(dirs):
        label = os.path.basename(d.rstrip("/"))
        pathname = d + ("/**/" if recursive else "/") + extension
        files = glob.iglob(pathname, recursive=recursive)
        files = sorted(filter(os.path.isfile, files))
        if process_fn is None:
            dataset[label].extend(files)
        else:
            dataset[label].extend(list(map(lambda x: process_fn(x), files)))

    return dataset


def split_dataset_train_test(
    dataset: pd.DataFrame, **train_test_split_args
) -> Tuple[DictDataset, DictDataset]:
    """Split dataset in train and test parts.

    Args:
        dataset: dict like dataset
        **train_test_split_args:
            test_size : float, int, or None (default is None)
                If float, should be between 0.0 and 1.0 and represent the
                proportion of the dataset to include in the test split. If
                int, represents the absolute number of test samples. If None,
                the value is automatically set
                to the complement of the train size.
                If train size is also None, test size is set to 0.25.

            train_size : float, int, or None (default is None)
                If float, should be between 0.0 and 1.0 and represent the
                proportion of the dataset to include in the train split. If
                int, represents the absolute number of train samples. If None,
                the value is automatically set
                to the complement of the test size.

            random_state : int or RandomState
                Pseudo-random number generator state used for random sampling.

            stratify : array-like or None (default is None)
                If not None, data is split in a stratified fashion,
                using this as the class labels.

    Returns:
        train and test dicts
    """
    train_dataset = defaultdict(list)
    test_dataset = defaultdict(list)
    for key, value in dataset.items():
        train_ids, test_ids = train_test_split(
            range(len(value)), **train_test_split_args
        )
        train_dataset[key].extend([value[i] for i in train_ids])
        test_dataset[key].extend([value[i] for i in test_ids])
    return train_dataset, test_dataset


def create_dataframe(dataset: DictDataset, **dataframe_args) -> pd.DataFrame:
    """Create pd.DataFrame from dict like `{key: [values]}`.

    Args:
        dataset: dict like `{key: [values]}`
        **dataframe_args:
            index : Index or array-like
                Index to use for resulting frame.
                Will default to np.arange(n) if no indexing information
                part of input data and no index provided
            columns : Index or array-like
                Column labels to use for resulting frame. Will default to
                np.arange(n) if no column labels are provided
            dtype : dtype, default None
                Data type to force, otherwise infer

    Returns:
        pd.DataFrame: dataframe from giving dataset
    """
    data = [
        (key, value) for key, values in dataset.items() for value in values
    ]
    df = pd.DataFrame(data, **dataframe_args)
    return df


__all__ = ["create_dataset", "create_dataframe", "split_dataset_train_test"]
