from typing import List, Dict, Optional, Tuple
import pandas as pd

from catalyst.utils.data import default_fold_split, stratified_fold_split, folds_to_list
from catalyst.utils.misc import args_are_not_none

from tqdm import tqdm
tqdm.pandas()


def map_dataframe(
        dataframe: pd.DataFrame,
        tag_column: str,
        class_column: str,
        tag2class: Dict[str, int],
        verbose: bool = False
) -> pd.DataFrame:
    """
    This function maps tags from ``tag_column`` to ints into ``class_column``
    Using ``tag2class`` dictionary

    Args:
        dataframe (pd.DataFrame): input dataframe
        tag_column (str): column with tags
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        (tuple): tuple with 4 dataframes
            whole dataframe, train part, valid part and infer part
    """

    if args_are_not_none(tag2class, tag_column, class_column):
        dataframe = map_dataframe(
            dataframe, tag_column, class_column, tag2class
        )

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

    return result_dataframe, df_train, df_valid, df_infer


def merge_multiple_fold_csv(
        fold_name: str,
        paths: Optional[str]
) -> pd.DataFrame:
    """
    Reads csv into one DataFrame with column ``fold``
    Args:
        fold_name: current fold name
        paths: paths to csv separated by commas
    Returns:
         pd.DataFrame: merged dataframes with column ``fold`` == ``fold_name``
    """
    result = pd.DataFrame()
    if paths is not None:
        for csv_path in paths.split(","):
            dataframe = pd.read_csv(csv_path)
            dataframe["fold"] = fold_name
            result = result.append(dataframe, ignore_index=True)

    return result


def parse_spec_csv(
    in_csv_train: str = None,
    in_csv_valid: str = None,
    in_csv_infer: str = None,
    tag2class: Optional[Dict[str, int]] = None,
    class_column: str = None,
    tag_column: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """This function reads train/valid/infer dataframes from giving paths
    Args:
        in_csv_train: paths to train csv separated by commas
        in_csv_valid: paths to valid csv separated by commas
        in_csv_infer: paths to infer csv separated by commas
        tag2class (Dict[str, int], optional): mapping from label names into ints
        tag_column (str, optional): column with label names
        class_column (str, optional): column to use for split
    Returns:
        (tuple): tuple with 4 dataframes
            whole dataframe, train part, valid part and infer part
    """
    df_train = merge_multiple_fold_csv(fold_name="train", paths=in_csv_train)
    df_valid = merge_multiple_fold_csv(fold_name="valid", paths=in_csv_valid)
    df_infer = merge_multiple_fold_csv(fold_name="infer", paths=in_csv_infer)

    if args_are_not_none(tag2class, tag_column, class_column):
        df_train = map_dataframe(df_train, tag_column, class_column, tag2class)
        df_valid = map_dataframe(df_valid, tag_column, class_column, tag2class)
        df_infer = map_dataframe(df_infer, tag_column, class_column, tag2class)

    result_dataframe = df_train.\
        append(df_valid, ignore_index=True).\
        append(df_infer, ignore_index=True)

    return result_dataframe, df_train, df_valid, df_infer
