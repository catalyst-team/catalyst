from typing import Union, List

import pandas as pd
import json
from catalyst.utils.data import default_fold_split, stratified_fold_split, dataframe_to_list, folds_to_list
from catalyst.utils.parse import merge_multiple_fold_csv

def parse_in_csvs(
    in_csv=None,
    in_csv_train=None,
    in_csv_valid=None,
    in_csv_infer=None,
    train_folds=None,
    valid_folds=None,
    tag2class=None,
    class_column=None,
    tag_column=None,
    folds_seed=42,
    n_folds=5
):
    in_csv_flag = in_csv is not None
    in_csv_spec_flag = (
        (in_csv_train is not None and in_csv_valid is not None)
        or in_csv_infer is not None
    )
    assert in_csv_flag != in_csv_spec_flag

    if in_csv_flag:
        df, df_train, df_valid, df_infer = parse_in_csv(
            in_csv=in_csv,
            train_folds=train_folds,
            valid_folds=valid_folds,
            tag2class=tag2class,
            class_column=class_column,
            tag_column=tag_column,
            seed=folds_seed,
            n_folds=n_folds
        )
    elif in_csv_spec_flag:
        df, df_train, df_valid, df_infer = parse_spec_csv(
            in_csv_train=in_csv_train,
            in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
            tag2class=tag2class,
            class_column=class_column,
            tag_column=tag_column
        )
    else:
        raise Exception("something go wrong")

    if len(df_train) > 0:
        del df_train["fold"]
        df_train = dataframe_to_list(df_train)

    if len(df_valid):
        del df_valid["fold"]
        df_valid = dataframe_to_list(df_valid)

    if len(df_infer) > 0:
        df_infer = dataframe_to_list(df_infer)

    return df, df_train, df_valid, df_infer
