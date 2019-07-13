import argparse
import json
import pandas as pd

from catalyst.utils.pandas import split_dataframe, folds_to_list


def build_args(parser):
    parser.add_argument(
        "--in-csv",
        type=str,
        dest="in_csv",
        help="Path to the csv to split",
        required=True
    )

    parser.add_argument(
        "-t",
        "--train-folds",
        type=str,
        dest="train_folds",
        help="Numbers separated by commas. They represent train folds",
        required=True
    )

    parser.add_argument(
        "-v",
        "--valid-folds",
        type=str,
        dest="valid_folds",
        default=None,
        help="Numbers separated by commas. They represent valid folds"
    )

    parser.add_argument(
        "-i",
        "--infer-folds",
        type=str,
        dest="infer_folds",
        default=None,
        help="Numbers separated by commas. They represent infer folds"
    )

    parser.add_argument(
        "--out-csv",
        type=str,
        dest="out_csv",
        help="Output CSV path for train and valid parts",
        required=True
    )

    parser.add_argument(
        "--tag2class",
        type=str,
        default=None,
        help="Path to YAML or JSON of label mappings"
    )

    parser.add_argument(
        "--tag-column",
        type=str,
        default=None,
        dest="tag_column",
        help="Column of labels (works in pair with `--tag2class` flag)"
    )

    parser.add_argument(
        "--class-column",
        type=str,
        default=None,
        dest="class_column",
        help="Column of classes"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for split folds"
    )

    parser.add_argument(
        "-n", "--n-folds", type=int, default=5, help="Number of result folds"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, uargs = parser.parse_known_args()
    return args, uargs


def main(args, uargs=None):
    dataframe = pd.read_csv(args.in_csv)

    train_folds = \
        folds_to_list(args.train_folds) \
        if args.train_folds is not None else None
    valid_folds = \
        folds_to_list(args.valid_folds) \
        if args.valid_folds is not None else None
    infer_folds = \
        folds_to_list(args.infer_folds) \
        if args.infer_folds is not None else None

    tag2class = json.load(open(args.tag2class)) \
        if args.tag2class is not None else None

    df_all, train, valid, infer = split_dataframe(
        dataframe,
        train_folds=train_folds,
        valid_folds=valid_folds,
        infer_folds=infer_folds,
        tag2class=tag2class,
        tag_column=args.tag_column,
        class_column=args.class_column,
        seed=args.seed,
        n_folds=args.n_folds
    )

    out_csv: str = args.out_csv
    if out_csv.endswith(".csv"):
        out_csv = out_csv[:-4]

    df_all.to_csv(f"{out_csv}.csv", index=False)
    train.to_csv(f"{out_csv}_train.csv", index=False)
    valid.to_csv(f"{out_csv}_valid.csv", index=False)
    infer.to_csv(f"{out_csv}_infer.csv", index=False)


if __name__ == "__main__":
    args, uargs = parse_args()
    main(args, uargs)
