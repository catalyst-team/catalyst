from typing import List, Dict, Optional, Tuple, Union
import pandas as pd

from catalyst.utils.dataset import \
    default_fold_split, stratified_fold_split
from catalyst.utils import args_are_not_none

from tqdm.auto import tqdm

tqdm.pandas()


def dataframe_to_list(dataframe: pd.DataFrame) -> List[dict]:
    """
    Converts dataframe to a list of rows (without indexes)

    Args:
        dataframe (DataFrame): input dataframe
    Returns:
        (List[dict]): list of rows
    """
    result = list(dataframe.to_dict(orient="index").values())
    return result


def folds_to_list(folds: Union[list, str, pd.Series]) -> List[int]:
    """
    This function formats string or either list of numbers
    into a list of unique int

    Args:
        folds (Union[list, str, pd.Series]): Either list of numbers or
            one string with numbers separated by commas or
            pandas series
    Returns:
        List[int]: list of unique ints
    Examples:
        >>> folds_to_list("1,2,1,3,4,2,4,6")
        [1, 2, 3, 4, 6]
        >>> folds_to_list([1, 2, 3.0, 5])
        [1, 2, 3, 5]
    Raises:
        ValueError: if value in string or array cannot be casted to int

    """
    if isinstance(folds, str):
        folds = folds.split(",")
    elif isinstance(folds, pd.Series):
        folds = list(sorted(folds.unique()))

    return list(sorted(list({int(x) for x in folds})))


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
            If none takes all folds not included in ``train_folds``
            and ``valid_folds``
        tag2class (Dict[str, int], optional): mapping from label names into int
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
    fold_name: str, paths: Optional[str]
) -> pd.DataFrame:
    """
    Reads csv into one DataFrame with column ``fold``
    Args:
        fold_name (str): current fold name
        paths (str): paths to csv separated by commas
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


def read_multiple_dataframes(
    in_csv_train: str = None,
    in_csv_valid: str = None,
    in_csv_infer: str = None,
    tag2class: Optional[Dict[str, int]] = None,
    class_column: str = None,
    tag_column: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """This function reads train/valid/infer dataframes from giving paths
    Args:
        in_csv_train (str): paths to train csv separated by commas
        in_csv_valid (str): paths to valid csv separated by commas
        in_csv_infer (str): paths to infer csv separated by commas
        tag2class (Dict[str, int], optional): mapping from label names into int
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

    result_dataframe = df_train. \
        append(df_valid, ignore_index=True). \
        append(df_infer, ignore_index=True)

    return result_dataframe, df_train, df_valid, df_infer


def read_csv_data(
    in_csv: str = None,
    train_folds: Optional[List[int]] = None,
    valid_folds: Optional[List[int]] = None,
    infer_folds: Optional[List[int]] = None,
    seed: int = 42,
    n_folds: int = 5,
    in_csv_train: str = None,
    in_csv_valid: str = None,
    in_csv_infer: str = None,
    tag2class: Optional[Dict[str, int]] = None,
    class_column: str = None,
    tag_column: str = None,
) -> Tuple[pd.DataFrame, List[dict], List[dict], List[dict]]:
    """
    From giving path ``in_csv`` reads a dataframe
    and split it to train/valid/infer folds
    or from several paths ``in_csv_train``, ``in_csv_valid``, ``in_csv_infer``
    reads independent folds.

    Note:
       This function can be used with different combinations of params.
        First block is used to get dataset from one `csv`:
            in_csv, train_folds, valid_folds, infer_folds, seed, n_folds
        Second includes paths to different csv for train/valid and infer parts:
            in_csv_train, in_csv_valid, in_csv_infer
        The other params (tag2class, tag_column, class_column) are optional
            for any previous block

    Args:
        in_csv (str): paths to whole dataset
        train_folds (List[int]): train folds
        valid_folds (List[int], optional): valid folds.
            If none takes all folds not included in ``train_folds``
        infer_folds (List[int], optional): infer folds.
            If none takes all folds not included in ``train_folds``
            and ``valid_folds``
        seed (int): seed for split
        n_folds (int): number of folds

        in_csv_train (str): paths to train csv separated by commas
        in_csv_valid (str): paths to valid csv separated by commas
        in_csv_infer (str): paths to infer csv separated by commas

        tag2class (Dict[str, int]): mapping from label names into ints
        tag_column (str): column with label names
        class_column (str): column to use for split

    Returns:
        (Tuple[pd.DataFrame, List[dict], List[dict], List[dict]]):
            tuple with 4 elements
            (whole dataframe,
            list with train data,
            list with valid data
            and list with infer data)
    """
    from_one_df: bool = in_csv is not None
    from_multiple_df: bool = \
        in_csv_train is not None \
        or in_csv_valid is not None \
        or in_csv_infer is not None

    if from_one_df == from_multiple_df:
        raise ValueError(
            "You should pass `in_csv` "
            "or `in_csv_train` with `in_csv_valid` but not both!"
        )

    if from_one_df:
        dataframe: pd.DataFrame = pd.read_csv(in_csv)
        dataframe, df_train, df_valid, df_infer = split_dataframe(
            dataframe,
            train_folds=train_folds,
            valid_folds=valid_folds,
            infer_folds=infer_folds,
            tag2class=tag2class,
            class_column=class_column,
            tag_column=tag_column,
            seed=seed,
            n_folds=n_folds
        )
    else:
        dataframe, df_train, df_valid, df_infer = read_multiple_dataframes(
            in_csv_train=in_csv_train,
            in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
            tag2class=tag2class,
            class_column=class_column,
            tag_column=tag_column
        )

    for data in [df_train, df_valid, df_infer]:
        if "fold" in data.columns:
            del data["fold"]

    result = (
        dataframe, dataframe_to_list(df_train), dataframe_to_list(df_valid),
        dataframe_to_list(df_infer)
    )

    return result
