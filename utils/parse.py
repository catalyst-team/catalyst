import pandas as pd
import json
from catalyst.utils.data import default_fold_split, stratified_fold_split, \
    split_dataframe


# ---- csv ----


def parse_csv2list(df):
    df = df.reset_index().drop("index", axis=1)
    df = list(df.to_dict("index").values())
    return df


def parse_in_csv(
        in_csv, train_folds, valid_folds=None,
        tag2class=None, class_column=None, tag_column=None,
        folds_seed=42, n_folds=5):
    df = pd.read_csv(in_csv)

    if tag2class is not None:
        with open(tag2class) as fin:
            cls2id = json.load(fin)
        df[class_column] = df[tag_column].apply(
            lambda x: cls2id[str(x)])

    if class_column is not None:
        df = stratified_fold_split(
            df,
            class_column=class_column,
            random_state=folds_seed,
            n_folds=n_folds)
    else:
        df = default_fold_split(
            df,
            random_state=folds_seed,
            n_folds=n_folds)

    train_folds = (
        train_folds
        if isinstance(train_folds, list)
        else list(map(int, train_folds.split(","))))
    df_train = df[df["fold"].isin(train_folds)]

    if valid_folds is not None:
        valid_folds = (
            valid_folds
            if isinstance(valid_folds, list)
            else list(map(int, valid_folds.split(","))))
        df_valid = df[df["fold"].isin(valid_folds)]
    else:
        df_valid = df[~df["fold"].isin(train_folds)]

    df_infer = []

    return df, df_train, df_valid, df_infer


def prepare_fold_csv(fold_name, **kwargs):
    spec_name = f"in_csv_{fold_name}"
    df = []
    if kwargs.get(spec_name, None) is not None:
        for csv in kwargs[spec_name].split(","):
            df_ = pd.read_csv(csv)
            df_["fold"] = fold_name
            df.append(df_)

    if len(df) > 0:
        df = pd.concat(df, axis=0)

    return df


def parse_spec_csv(
        in_csv_train=None, in_csv_valid=None, in_csv_infer=None,
        tag2class=None, class_column=None, tag_column=None):
    df_train = prepare_fold_csv(
        fold_name="train",
        in_csv_train=in_csv_train,
        in_csv_valid=in_csv_valid,
        in_csv_infer=in_csv_infer)
    df_valid = prepare_fold_csv(
        fold_name="valid",
        in_csv_train=in_csv_train,
        in_csv_valid=in_csv_valid,
        in_csv_infer=in_csv_infer)
    df_infer = prepare_fold_csv(
        fold_name="infer",
        in_csv_train=in_csv_train,
        in_csv_valid=in_csv_valid,
        in_csv_infer=in_csv_infer)

    if len(df_train) > 0 and len(df_valid) > 0:
        df = pd.concat([df_train, df_valid], axis=0)
    else:
        df = []

    if tag2class is not None:
        with open(tag2class) as fin:
            cls2id = json.load(fin)
        if len(df) > 0:
            df[class_column] = df[tag_column].apply(
                lambda x: cls2id[str(x)])
        if len(df_train) > 0:
            df_train[class_column] = df_train[tag_column].apply(
                lambda x: cls2id[str(x)])
        if len(df_valid) > 0:
            df_valid[class_column] = df_valid[tag_column].apply(
                lambda x: cls2id[str(x)])
        if len(df_infer) > 0:
            df_infer[class_column] = df_infer[tag_column].apply(
                lambda x: cls2id[str(x)])

    return df, df_train, df_valid, df_infer


def parse_in_csvs(
        in_csv=None, in_csv_train=None, in_csv_valid=None, in_csv_infer=None,
        train_folds=None, valid_folds=None,
        tag2class=None, class_column=None, tag_column=None,
        folds_seed=42, n_folds=5):
    in_csv_flag = in_csv is not None
    in_csv_spec_flag = (
            (in_csv_train is not None and in_csv_valid is not None)
            or in_csv_infer is not None)
    assert in_csv_flag != in_csv_spec_flag

    if in_csv_flag:
        df, df_train, df_valid, df_infer = parse_in_csv(
            in_csv=in_csv, train_folds=train_folds, valid_folds=valid_folds,
            tag2class=tag2class,
            class_column=class_column, tag_column=tag_column,
            folds_seed=folds_seed, n_folds=n_folds)
    elif in_csv_spec_flag:
        df, df_train, df_valid, df_infer = parse_spec_csv(
            in_csv_train=in_csv_train, in_csv_valid=in_csv_valid,
            in_csv_infer=in_csv_infer,
            tag2class=tag2class,
            class_column=class_column, tag_column=tag_column)
    else:
        raise Exception("something go wrong")

    if len(df_train) > 0:
        del df_train["fold"]
        df_train = parse_csv2list(df_train)

    if len(df_valid):
        del df_valid["fold"]
        df_valid = parse_csv2list(df_valid)

    if len(df_infer) > 0:
        df_infer = parse_csv2list(df_infer)

    return df, df_train, df_valid, df_infer


# ---- txt ----


def read_in_txt(filepath):
    with open(filepath) as fin:
        data = fin.readlines()
    data = list(map(lambda x: x.replace("\n", ""), data))
    return data


def parse_in_txt(data_params):
    data = read_in_txt(data_params["in_txt"])
    data_train, data_valid = split_dataframe(
        data, test_size=0.2, random_state=42)
    return data, data_train, data_valid, []


def parse_spec_txt(data_params):
    data_train = (
        read_in_txt(data_params["in_txt_train"])
        if data_params.get("in_txt_train", None) is not None
        else [])
    data_valid = (
        read_in_txt(data_params["in_txt_valid"])
        if data_params.get("in_txt_valid", None) is not None
        else [])
    data_infer = (
        read_in_txt(data_params["in_txt_infer"])
        if data_params.get("in_txt_infer", None) is not None
        else [])
    return [], data_train, data_valid, data_infer


def parse_in_txts(data_params):
    in_txt_flag = data_params.get("in_txt", None) is not None
    in_txt_spec_flag = (
            (data_params.get("in_txt_train", None) is not None
             and data_params.get("in_txt_valid", None) is not None)
            or data_params.get("in_txt_infer", None) is not None)
    assert in_txt_flag != in_txt_spec_flag

    if in_txt_flag:
        df, df_train, df_valid, df_infer = parse_in_txt(data_params)
    elif in_txt_spec_flag:
        df, df_train, df_valid, df_infer = parse_spec_txt(data_params)
    else:
        raise Exception("something go wrong")

    return df, df_train, df_valid, df_infer
