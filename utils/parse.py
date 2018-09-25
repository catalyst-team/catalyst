import pandas as pd
import json
from prometheus.utils.data import default_fold_split, stratified_fold_split, \
    split_dataframe


# ---- csv ----


def parse_csv2list(df):
    df = df.reset_index().drop("index", axis=1)
    df = list(df.to_dict("index").values())
    return df


def parse_in_csv(data_params):
    df = pd.read_csv(data_params["in_csv"])

    if "tag2class" in data_params:
        with open(data_params["tag2class"]) as fin:
            cls2id = json.load(fin)
        class_column = data_params["class_column"]
        tag_column = data_params["tag_column"]
        df[class_column] = df[tag_column].apply(
            lambda x: cls2id[str(x)])

    if data_params.get("class_column", None) is not None:
        df = stratified_fold_split(
            df,
            class_column=data_params["class_column"],
            random_state=data_params.get("folds_seed", 42),
            n_folds=data_params["n_folds"])
    else:
        df = default_fold_split(
            df,
            random_state=data_params.get("folds_seed", 42),
            n_folds=data_params["n_folds"])

    train_folds = (
        data_params["train_folds"]
        if isinstance(data_params["train_folds"], list)
        else list(map(int, data_params["train_folds"].split(","))))
    df_train = df[df["fold"].isin(train_folds)]

    if "valid_folds" in data_params:
        valid_folds = (
            data_params["valid_folds"]
            if isinstance(data_params["valid_folds"], list)
            else list(map(int, data_params["valid_folds"].split(","))))
        df_valid = df[df["fold"].isin(valid_folds)]
    else:
        df_valid = df[~df["fold"].isin(train_folds)]

    df_infer = []

    return df, df_train, df_valid, df_infer


def prepare_fold_csv(data_params, fold_name):
    spec_name = f"in_csv_{fold_name}"
    df = []
    if data_params.get(spec_name, None) is not None:
        for csv in data_params[spec_name].split(","):
            df_ = pd.read_csv(csv)
            df_["fold"] = fold_name
            df.append(df_)

    if len(df) > 0:
        df = pd.concat(df, axis=0)

    return df


def parse_spec_csv(data_params):
    df_train = prepare_fold_csv(data_params, "train")
    df_valid = prepare_fold_csv(data_params, "valid")
    df_infer = prepare_fold_csv(data_params, "infer")

    if len(df_train) > 0 and len(df_valid) > 0:
        df = pd.concat([df_train, df_valid], axis=0)
    else:
        df = []

    if data_params.get("tag2class", None) is not None:
        with open(data_params["tag2class"]) as fin:
            cls2id = json.load(fin)
        class_column = data_params["class_column"]
        tag_column = data_params["tag_column"]
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


def parse_in_csvs(data_params):
    in_csv_flag = data_params.get("in_csv", None) is not None
    in_csv_spec_flag = (
            (data_params.get("in_csv_train", None) is not None
             and data_params.get("in_csv_valid", None) is not None)
            or data_params.get("in_csv_infer", None) is not None)
    assert in_csv_flag != in_csv_spec_flag

    if in_csv_flag:
        df, df_train, df_valid, df_infer = parse_in_csv(data_params)
    elif in_csv_spec_flag:
        df, df_train, df_valid, df_infer = parse_spec_csv(data_params)
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
