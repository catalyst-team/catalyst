from collections import namedtuple
from typing import List, Dict, Optional

import pandas as pd

from catalyst.utils.data import default_fold_split, stratified_fold_split, folds_to_list
from catalyst.utils.misc import args_are_not_none


SplitDataFrame = namedtuple(
    "SplitDataFrame", ["whole", "train", "valid", "infer"]
)


def split_dataframe(
        dataframe: pd.DataFrame,
        train_folds: List[int],
        valid_folds: Optional[List[int]] = None,
        infer_folds: Optional[List[int]] = None,
        tag2class: Optional[Dict[str, int]] = None,
        tag_column: str = None,
        class_column: str = None,
        seed: int = 42,
        n_folds: int = 5
) -> SplitDataFrame:
    """
    Split a Pandas DataFrame into folds.
    Args:
        dataframe (pd.DataFrame): input dataframe
        train_folds (List[int]): train folds
        valid_folds (List[int], optional): valid folds.
            If none takes all folds not included in ``train_folds``
        infer_folds (List[int], optional): infer folds.
            If none takes all folds not included in ``train_folds`` and ``valid_folds``
        tag2class (Dict[str, int], optional): mapping from label names into ints
        tag_column (str, optional): column with label names
        class_column (str, optional): column to use for split
        seed (int): seed for split
        n_folds (int): number of folds
    Returns:
        (SplitDataFrame): tuple with whole dataframe, train part, valid part and infer part
    """

    if args_are_not_none(tag2class, tag_column, class_column):
        def map_label(x):
            return tag2class[str(x)]

        dataframe[class_column] = dataframe[tag_column].apply(map_label)

    if class_column is not None:
        result_dataframe = stratified_fold_split(
            dataframe,
            class_column=class_column,
            random_state=seed,
            n_folds=n_folds
        )
    else:
        result_dataframe = default_fold_split(
            dataframe, random_state=seed, n_folds=n_folds
        )

    fold_series = result_dataframe["fold"]

    train_folds = folds_to_list(train_folds)
    df_train = result_dataframe[fold_series.isin(train_folds)]

    if valid_folds is None:
        mask = ~fold_series.isin(train_folds)
        valid_folds = result_dataframe[mask]["fold"]

    valid_folds = folds_to_list(valid_folds)
    df_valid = result_dataframe[fold_series.isin(valid_folds)]

    infer_folds = folds_to_list(infer_folds or [])
    df_infer = result_dataframe[fold_series.isin(infer_folds)]

    return SplitDataFrame(result_dataframe, df_train, df_valid, df_infer)
