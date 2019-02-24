import argparse

import pandas as pd
import safitty

from catalyst.utils.data import folds_to_list
from catalyst.utils.parse import SplitDataFrame, split_dataframe


def build_args(parser):
    parser.add_argument(
        "--in-csv",
        type=str,
        dest="in_csv",
        help="Path to the csv to split",
        required=True
    )

    parser.add_argument(
        "-t", "--train-folds",
        type=str,
        dest="train_folds",
        help="Integers separated by commas. This is represents train folds",
        required=True
    )

    parser.add_argument(
        "--out-csv",
        type=str,
        dest="out_csv",
        help="Output CSV path for train and valid parts",
        required=True
    )

    parser.add_argument(
        "-m",
        "--label-mapping",
        type=str,
        default=None,
        dest="label_mapping",
        help="Path to YAML or JSON of label mappings"
    )

    parser.add_argument(
        "-l",
        "--label-column",
        type=str,
        default=None,
        dest="label_column",
        help="Column of labels (works in pair with `--label-mapping` flag)"
    )

    parser.add_argument(
        "-c",
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
        "-n", "--n_folds", type=int, default=5, help="Number of result folds"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args):
    dataframe = pd.read_csv(args.in_csv)
    train_folds = folds_to_list(args.train_folds)

    if args.valid_folds is not None:
        valid_folds = folds_to_list(args.valid_folds)
    else:
        valid_folds = None

    if args.label_mapping is not None:
        label_mapping = safitty.load_config(args.label_mapping)
    else:
        label_mapping = None

    result: SplitDataFrame = split_dataframe(
        dataframe,
        train_folds=train_folds,
        valid_folds=valid_folds,
        label_mapping=label_mapping,
        label_column=args.label_column,
        class_column=args.class_column,
        seed=args.seed,
        n_folds=args.n_folds
    )

    out_csv: str = args.out_csv
    if out_csv.endswith(".csv"):
        out_csv = out_csv[:-4]

    result.train.to_csv(f"{out_csv}_train.csv", index=False)
    result.valid.to_csv(f"{out_csv}_valid.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
