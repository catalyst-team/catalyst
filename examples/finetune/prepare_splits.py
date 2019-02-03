import argparse
import pandas as pd
from catalyst.legacy.utils.parse import parse_in_csvs


def build_args(parser):
    parser.add_argument("--in-csv", type=str, required=True)
    parser.add_argument("--tag2class", type=str, required=True)
    parser.add_argument("--tag-column", type=str, required=True)
    parser.add_argument("--class-column", type=str, required=True)
    parser.add_argument("--folds-seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)

    parser.add_argument("--train-folds", type=str, default=None)
    parser.add_argument("--valid-folds", type=str, default=None)

    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--out-csv-train", type=str, default=None)
    parser.add_argument("--out-csv-valid", type=str, default=None)
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    train_folds = (
        list(map(int, args.train_folds.split(",")))
        if args.train_folds is not None else None
    )
    valid_folds = (
        list(map(int, args.valid_folds.split(",")))
        if args.valid_folds is not None else None
    )

    df, df_train, df_valid, _ = parse_in_csvs(
        in_csv=args.in_csv,
        tag2class=args.tag2class,
        tag_column=args.tag_column,
        class_column=args.class_column,
        train_folds=train_folds,
        valid_folds=valid_folds,
        folds_seed=args.folds_seed,
        n_folds=args.n_folds
    )
    df.to_csv(args.out_csv, index=False)

    if args.out_csv_train is not None:
        df_train = pd.DataFrame(df_train)
        df_train.to_csv(args.out_csv_train, index=False)
    if args.out_csv_valid is not None:
        df_valid = pd.DataFrame(df_valid)
        df_valid.to_csv(args.out_csv_valid, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
