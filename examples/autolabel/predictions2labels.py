import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd


def build_args(parser):
    parser.add_argument("--in-npy", type=str, required=True)
    parser.add_argument("--in-csv-infer", type=str, required=True)
    parser.add_argument("--in-csv-train", type=str, required=True)
    parser.add_argument("--in-tag2cls", type=str, required=True)
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


def path2name(x):
    return x.rsplit("/", 1)[-1]


def main(args, _=None):
    logits = np.load(args.in_npy, mmap_mode="r")
    probs = softmax(logits)
    confidence = np.max(probs, axis=1)
    preds = np.argmax(logits, axis=1)

    df_infer = pd.read_csv(args.in_csv_infer)
    df_train = pd.read_csv(args.in_csv_train)

    df_infer["filename"] = df_infer["filepath"].apply(path2name)
    df_train["filename"] = df_train["filepath"].apply(path2name)

    with open(args.in_tag2cls) as fin:
        tag2lbl = json.load(fin)
        cls2tag = {int(v): k for k, v in tag2lbl.items()}

    preds = [cls2tag[x] for x in preds]
    df_infer["tag"] = preds
    df_infer["confidence"] = confidence

    train_filepath = df_train["filename"].tolist()
    df_infer = df_infer[~df_infer["filename"].isin(train_filepath)]

    counter_ = 0
    for i, row in df_infer.iterrows():
        if row["confidence"] < args.threshold:
            continue
        filepath_src = f"{args.in_dir}/{row['filepath']}"
        filename = filepath_src.rsplit("/", 1)[-1]
        filepath_dst = f"{args.out_dir}/{row['tag']}/{filename}"
        folder_dst = filepath_dst.rsplit("/", 1)[0]
        os.makedirs(folder_dst, exist_ok=True)
        shutil.copy2(filepath_src, filepath_dst)
        counter_ += 1
    print(f"Predicted: {counter_} ({100*counter_/len(df_infer):2.2f}%)")


if __name__ == '__main__':
    args = parse_args()
    main(args)
