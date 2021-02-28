# flake8: noqa
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import glob
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle

from catalyst.utils.misc import args_are_not_none

DictDataset = Dict[str, List[Any]]


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
        dirs: path to dirs, for example /home/user/data/**
        extension: data extension you are looking for
        process_fn (Callable[[str], object]): function(path_to_file) -> object
            process function for found files, by default
        recursive: enables recursive globbing

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
            dataset[label].extend([process_fn(x) for x in files])

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
        train_ids, test_ids = train_test_split(range(len(value)), **train_test_split_args)
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
    data = [(key, value) for key, values in dataset.items() for value in values]
    df = pd.DataFrame(data, **dataframe_args)
    return df


def dataframe_to_list(dataframe: pd.DataFrame) -> List[dict]:
    """Converts dataframe to a list of rows (without indexes).

    Args:
        dataframe: input dataframe

    Returns:
        List[dict]: list of rows
    """
    result = list(dataframe.to_dict(orient="index").values())
    return result


def folds_to_list(folds: Union[list, str, pd.Series]) -> List[int]:
    """This function formats string or either list of numbers
    into a list of unique int.

    Examples:
        >>> folds_to_list("1,2,1,3,4,2,4,6")
        [1, 2, 3, 4, 6]
        >>> folds_to_list([1, 2, 3.0, 5])
        [1, 2, 3, 5]

    Args:
        folds (Union[list, str, pd.Series]): Either list of numbers or
            one string with numbers separated by commas or
            pandas series

    Returns:
        List[int]: list of unique ints

    Raises:
        ValueError: if value in string or array cannot be casted to int
    """
    if isinstance(folds, str):
        folds = folds.split(",")
    elif isinstance(folds, pd.Series):
        folds = sorted(folds.unique())

    return sorted({int(x) for x in folds})


def split_dataframe_train_test(
    dataframe: pd.DataFrame, **train_test_split_args
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe in train and test part.

    Args:
        dataframe: pd.DataFrame to split
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
        train and test DataFrames

    .. note::
        It exist cause sklearn `split` is overcomplicated.
    """
    df_train, df_test = train_test_split(dataframe, **train_test_split_args)
    return df_train, df_test


def split_dataframe_on_folds(
    dataframe: pd.DataFrame, random_state: int = 42, n_folds: int = 5
) -> pd.DataFrame:
    """Splits DataFrame into `N` folds.

    Args:
        dataframe: a dataset
        random_state: seed for random shuffle
        n_folds: number of result folds

    Returns:
        pd.DataFrame: new dataframe with `fold` column
    """
    dataframe = shuffle(dataframe, random_state=random_state)

    df_tmp = []
    for i, df_el in enumerate(np.array_split(dataframe, n_folds)):
        df_el["fold"] = i
        df_tmp.append(df_el)
    dataframe = pd.concat(df_tmp)
    return dataframe


def split_dataframe_on_stratified_folds(
    dataframe: pd.DataFrame, class_column: str, random_state: int = 42, n_folds: int = 5,
) -> pd.DataFrame:
    """Splits DataFrame into `N` stratified folds.

    Also see :class:`catalyst.data.sampler.BalanceClassSampler`

    Args:
        dataframe: a dataset
        class_column: which column to use for split
        random_state: seed for random shuffle
        n_folds: number of result folds

    Returns:
        pd.DataFrame: new dataframe with `fold` column
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_column = np.zeros(len(dataframe), dtype=int)
    for i, (_, test_index) in enumerate(skf.split(range(len(dataframe)), dataframe[class_column])):
        fold_column[test_index] = i
    dataframe["fold"] = fold_column
    return dataframe


def split_dataframe_on_column_folds(
    dataframe: pd.DataFrame, column: str, random_state: int = 42, n_folds: int = 5,
) -> pd.DataFrame:
    """Splits DataFrame into `N` folds.

    Args:
        dataframe: a dataset
        column: which column to use
        random_state: seed for random shuffle
        n_folds: number of result folds

    Returns:
        pd.DataFrame: new dataframe with `fold` column
    """
    df_tmp = []
    labels = shuffle(sorted(dataframe[column].unique()), random_state=random_state)
    for i, fold_labels in enumerate(np.array_split(labels, n_folds)):
        df_label = dataframe[dataframe[column].isin(fold_labels)]
        df_label["fold"] = i
        df_tmp.append(df_label)
    dataframe = pd.concat(df_tmp)
    return dataframe


def map_dataframe(
    dataframe: pd.DataFrame,
    tag_column: str,
    class_column: str,
    tag2class: Dict[str, int],
    verbose: bool = False,
) -> pd.DataFrame:
    """This function maps tags from ``tag_column`` to ints into
    ``class_column`` using ``tag2class`` dictionary.

    Args:
        dataframe: input dataframe
        tag_column: column with tags
        class_column (str) output column with classes
        tag2class (Dict[str, int]): mapping from tags to class labels
        verbose: flag if true, uses tqdm

    Returns:
        pd.DataFrame: updated dataframe with ``class_column``
    """
    dataframe: pd.DataFrame = dataframe.copy()

    def map_label(x):
        return tag2class[str(x)]

    if verbose:
        series: pd.Series = dataframe[tag_column].progress_apply(map_label)
    else:
        series: pd.Series = dataframe[tag_column].apply(map_label)

    dataframe.loc[series.index, class_column] = series
    return dataframe


def separate_tags(
    dataframe: pd.DataFrame, tag_column: str = "tag", tag_delim: str = ","
) -> pd.DataFrame:
    """Separates values in ``class_column`` column.

    Args:
        dataframe: a dataset
        tag_column: column name to separate values
        tag_delim: delimiter to separate values

    Returns:
        pd.DataFrame: new dataframe
    """
    df_new = []
    for _, row in dataframe.iterrows():
        for class_name in row[tag_column].split(tag_delim):
            df_new.append({**row, **{tag_column: class_name}})
    df_new = pd.DataFrame(df_new)
    return df_new


def get_dataset_labeling(dataframe: pd.DataFrame, tag_column: str) -> Dict[str, int]:
    """Prepares a mapping using unique values from ``tag_column``.

    .. code-block:: javascript

        {
            "class_name_0": 0,
            "class_name_1": 1,
            ...
            "class_name_N": N
        }

    Args:
        dataframe: a dataset
        tag_column: which column to use

    Returns:
        Dict[str, int]: mapping from tag to labels
    """
    tag_to_labels = {
        str(class_name): label
        for label, class_name in enumerate(sorted(dataframe[tag_column].unique()))
    }
    return tag_to_labels


def split_dataframe(
    dataframe: pd.DataFrame,
    train_folds: List[int],
    valid_folds: Optional[List[int]] = None,
    infer_folds: Optional[List[int]] = None,
    tag2class: Optional[Dict[str, int]] = None,
    tag_column: str = None,
    class_column: str = None,
    seed: int = 42,
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a Pandas DataFrame into folds.

    Args:
        dataframe: input dataframe
        train_folds: train folds
        valid_folds (List[int], optional): valid folds.
            If none takes all folds not included in ``train_folds``
        infer_folds (List[int], optional): infer folds.
            If none takes all folds not included in ``train_folds``
            and ``valid_folds``
        tag2class (Dict[str, int], optional): mapping from label names into int
        tag_column (str, optional): column with label names
        class_column (str, optional): column to use for split
        seed: seed for split
        n_folds: number of folds

    Returns:
        tuple: tuple with 4 dataframes
            whole dataframe, train part, valid part and infer part
    """
    if args_are_not_none(tag2class, tag_column, class_column):
        dataframe = map_dataframe(dataframe, tag_column, class_column, tag2class)

    if class_column is not None:
        df_all = split_dataframe_on_stratified_folds(
            dataframe, class_column=class_column, random_state=seed, n_folds=n_folds,
        )
    else:
        df_all = split_dataframe_on_folds(dataframe, random_state=seed, n_folds=n_folds)

    fold_series = df_all["fold"]

    train_folds = folds_to_list(train_folds)
    df_train = df_all[fold_series.isin(train_folds)]

    if valid_folds is None:
        mask = ~fold_series.isin(train_folds)
        valid_folds = df_all[mask]["fold"]

    valid_folds = folds_to_list(valid_folds)
    df_valid = df_all[fold_series.isin(valid_folds)]

    infer_folds = folds_to_list(infer_folds or [])
    df_infer = df_all[fold_series.isin(infer_folds)]

    return df_all, df_train, df_valid, df_infer


# def merge_multiple_fold_csv(fold_name: str, paths: Optional[str]) -> pd.DataFrame:
#     """Reads csv into one DataFrame with column ``fold``.
#
#     Args:
#         fold_name: current fold name
#         paths: paths to csv separated by commas
#
#     Returns:
#          pd.DataFrame: merged dataframes with column ``fold`` == ``fold_name``
#     """
#     result = pd.DataFrame()
#     if paths is not None:
#         for csv_path in paths.split(","):
#             dataframe = pd.read_csv(csv_path)
#             dataframe["fold"] = fold_name
#             result = result.append(dataframe, ignore_index=True)
#
#     return result


# def read_multiple_dataframes(
#     in_csv_train: str = None,
#     in_csv_valid: str = None,
#     in_csv_infer: str = None,
#     tag2class: Optional[Dict[str, int]] = None,
#     class_column: str = None,
#     tag_column: str = None,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """This function reads train/valid/infer dataframes from giving paths.
#
#     Args:
#         in_csv_train: paths to train csv separated by commas
#         in_csv_valid: paths to valid csv separated by commas
#         in_csv_infer: paths to infer csv separated by commas
#         tag2class (Dict[str, int], optional): mapping from label names into int
#         tag_column (str, optional): column with label names
#         class_column (str, optional): column to use for split
#
#     Returns:
#         tuple: tuple with 4 dataframes
#             whole dataframe, train part, valid part and infer part
#     """
#     assert any(x is not None for x in (in_csv_train, in_csv_valid, in_csv_infer))
#
#     result_df = None
#     fold_dfs = {}
#     for fold_df, fold_name in zip(
#         (in_csv_train, in_csv_valid, in_csv_infer), ("train", "valid", "infer")
#     ):
#         if fold_df is not None:
#             fold_df = merge_multiple_fold_csv(fold_name=fold_name, paths=fold_df)
#             if args_are_not_none(tag2class, tag_column, class_column):
#                 fold_df = map_dataframe(fold_df, tag_column, class_column, tag2class)
#             fold_dfs[fold_name] = fold_df
#
#             result_df = (
#                 fold_df if result_df is None else result_df.append(fold_df, ignore_index=True)
#             )
#
#     output = (
#         result_df,
#         fold_dfs.get("train", None),
#         fold_dfs.get("valid", None),
#         fold_dfs.get("infer", None),
#     )
#
#     return output


# def read_csv_data(
#     in_csv: str = None,
#     train_folds: Optional[List[int]] = None,
#     valid_folds: Optional[List[int]] = None,
#     infer_folds: Optional[List[int]] = None,
#     seed: int = 42,
#     n_folds: int = 5,
#     in_csv_train: str = None,
#     in_csv_valid: str = None,
#     in_csv_infer: str = None,
#     tag2class: Optional[Dict[str, int]] = None,
#     class_column: str = None,
#     tag_column: str = None,
# ) -> Tuple[pd.DataFrame, List[dict], List[dict], List[dict]]:
#     """
#     From giving path ``in_csv`` reads a dataframe
#     and split it to train/valid/infer folds
#     or from several paths ``in_csv_train``, ``in_csv_valid``, ``in_csv_infer``
#     reads independent folds.
#
#     .. note::
#        This function can be used with different combinations of params.
#         First block is used to get dataset from one `csv`:
#             in_csv, train_folds, valid_folds, infer_folds, seed, n_folds
#         Second includes paths to different csv for train/valid and infer parts:
#             in_csv_train, in_csv_valid, in_csv_infer
#         The other params (tag2class, tag_column, class_column) are optional
#             for any previous block
#
#     Args:
#         in_csv: paths to whole dataset
#         train_folds: train folds
#         valid_folds (List[int], optional): valid folds.
#             If none takes all folds not included in ``train_folds``
#         infer_folds (List[int], optional): infer folds.
#             If none takes all folds not included in ``train_folds``
#             and ``valid_folds``
#         seed: seed for split
#         n_folds: number of folds
#
#         in_csv_train: paths to train csv separated by commas
#         in_csv_valid: paths to valid csv separated by commas
#         in_csv_infer: paths to infer csv separated by commas
#
#         tag2class (Dict[str, int]): mapping from label names into ints
#         tag_column: column with label names
#         class_column: column to use for split
#
#     Returns:
#         Tuple[pd.DataFrame, List[dict], List[dict], List[dict]]:
#             tuple with 4 elements
#             (whole dataframe,
#             list with train data,
#             list with valid data
#             and list with infer data)
#     """
#     from_one_df: bool = in_csv is not None
#     from_multiple_df: bool = (
#         in_csv_train is not None or in_csv_valid is not None or in_csv_infer is not None
#     )
#
#     if from_one_df == from_multiple_df:
#         raise ValueError(
#             "You should pass `in_csv` " "or `in_csv_train` with `in_csv_valid` but not both!"
#         )
#
#     if from_one_df:
#         dataframe: pd.DataFrame = pd.read_csv(in_csv)
#         dataframe, df_train, df_valid, df_infer = split_dataframe(
#             dataframe,
#             train_folds=train_folds,
#             valid_folds=valid_folds,
#             infer_folds=infer_folds,
#             tag2class=tag2class,
#             class_column=class_column,
#             tag_column=tag_column,
#             seed=seed,
#             n_folds=n_folds,
#         )
#     else:
#         dataframe, df_train, df_valid, df_infer = read_multiple_dataframes(
#             in_csv_train=in_csv_train,
#             in_csv_valid=in_csv_valid,
#             in_csv_infer=in_csv_infer,
#             tag2class=tag2class,
#             class_column=class_column,
#             tag_column=tag_column,
#         )
#
#     for data in [df_train, df_valid, df_infer]:
#         if data is not None and "fold" in data.columns:
#             del data["fold"]
#
#     result = (
#         dataframe,
#         dataframe_to_list(df_train) if df_train is not None else None,
#         dataframe_to_list(df_valid) if df_valid is not None else None,
#         dataframe_to_list(df_infer) if df_infer is not None else None,
#     )
#
#     return result


def balance_classes(
    dataframe: pd.DataFrame,
    class_column: str = "label",
    random_state: int = 42,
    how: str = "downsampling",
) -> pd.DataFrame:
    """Balance classes in dataframe by ``class_column``.

    See also :class:`catalyst.data.sampler.BalanceClassSampler`.

    Args:
        dataframe: a dataset
        class_column: which column to use for split
        random_state: seed for random shuffle
        how: strategy to sample, must be one on ["downsampling", "upsampling"]

    Returns:
        pd.DataFrame: new dataframe with balanced ``class_column``

    Raises:
        NotImplementedError:
            if `how` is not in ["upsampling", "downsampling", int]
    """
    cnt = defaultdict(lambda: 0.0)
    for label in sorted(dataframe[class_column].unique()):
        cnt[label] = len(dataframe[dataframe[class_column] == label])

    if isinstance(how, int) or how == "upsampling":
        samples_per_class = how if isinstance(how, int) else max(cnt.values())

        balanced_dfs = {}
        for label in sorted(dataframe[class_column].unique()):
            df_class_column = dataframe[dataframe[class_column] == label]
            if samples_per_class <= len(df_class_column):
                balanced_dfs[label] = df_class_column.sample(
                    samples_per_class, replace=True, random_state=random_state
                )
            else:
                df_class_column_additional = df_class_column.sample(
                    samples_per_class - len(df_class_column),
                    replace=True,
                    random_state=random_state,
                )
                balanced_dfs[label] = pd.concat((df_class_column, df_class_column_additional))
    elif how == "downsampling":
        samples_per_class = min(cnt.values())

        balanced_dfs = {}
        for label in sorted(dataframe[class_column].unique()):
            balanced_dfs[label] = dataframe[dataframe[class_column] == label].sample(
                samples_per_class, replace=False, random_state=random_state
            )
    else:
        raise NotImplementedError()

    balanced_df = pd.concat(balanced_dfs.values())

    return balanced_df


__all__ = [
    "dataframe_to_list",
    "folds_to_list",
    "split_dataframe",
    "split_dataframe_on_column_folds",
    "split_dataframe_on_folds",
    "split_dataframe_on_stratified_folds",
    "split_dataframe_train_test",
    "separate_tags",
    "map_dataframe",
    "get_dataset_labeling",
    "balance_classes",
    "create_dataset",
    "create_dataframe",
    "split_dataset_train_test",
]
