from collections import namedtuple
from typing import List, Dict, Optional

import pandas as pd

from catalyst.utils.data import default_fold_split, stratified_fold_split
from catalyst.utils.misc import args_are_not_none


SplitDataFrame = namedtuple(
    "SplitDataFrame", ["whole", "train", "valid", "infer"]
)


def split_dataframe(
        dataframe: pd.DataFrame,
        train_folds: List[int],
        valid_folds: Optional[List[int]] = None,
        label_mapping: Optional[Dict[str, int]] = None,
        label_column: str = None,
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
        label_mapping (Dict[str, int], optional): mapping from label names into ints
        label_column (str, optional): column with label names
        class_column (str, optional): column to use for split
        seed (int): seed for split
        n_folds (int): number of folds
    Returns:
        (SplitDataFrame): tuple with whole dataframe, train part, valid part and empty infer part
    """

    if args_are_not_none(label_mapping, label_column, class_column):
        def map_label(x):
            return label_mapping[str(x)]

        dataframe[class_column] = dataframe[label_column].apply(map_label)

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

    df_train = result_dataframe[result_dataframe["fold"].isin(train_folds)]

    if valid_folds is not None:
        df_valid = result_dataframe[result_dataframe["fold"].isin(valid_folds)]
    else:
        df_valid = result_dataframe[~result_dataframe["fold"].isin(train_folds)]

    return SplitDataFrame(result_dataframe, df_train, df_valid, pd.DataFrame())
