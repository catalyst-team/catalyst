import argparse
import json
import pandas as pd

from catalyst.utils.dataset import create_dataset, create_dataframe, \
    prepare_dataset_labeling, separate_tags


def prepare_df_from_dirs(in_dirs, tag_column_name):
    dfs = []
    splitted_dirs = in_dirs.strip(",").split(",")

    def process_fn(x):
        if len(splitted_dirs) == 1:
            # remove all in_dir part from path
            return x.replace(f"{in_dir}", "")
        else:
            # leaves last part of in_dir path,
            #  which identifies separate in_dir
            return x.replace(f"{in_dir}", f"{in_dir.split('/')[-2]}/")

    for in_dir in splitted_dirs:
        if not in_dir.endswith("/"):
            in_dir = f"{in_dir}/"

        dataset = create_dataset(f"{in_dir}/**", process_fn=process_fn)

        dfs.append(
            create_dataframe(dataset, columns=[tag_column_name, "filepath"])
        )

    df = pd.concat(dfs).reset_index(drop=True)
    return df


def build_args(parser):
    parser.add_argument(
        "--in-csv", type=str, default=None, help="Path to data in `.csv`."
    )
    parser.add_argument(
        "--in-dir",
        type=str,
        default=None,
        help="Path to directory with dataset"
        "or paths separated by commas for several datasets"
    )

    parser.add_argument(
        "--out-dataset",
        type=str,
        default=None,
        required=True,
        help="Path to output dataframe"
    )
    parser.add_argument(
        "--out-labeling",
        type=str,
        default=None,
        required=True,
        help="Path to output JSON"
    )

    parser.add_argument(
        "--tag-column", type=str, default="tag", help="Target column name"
    )
    parser.add_argument(
        "--tag-delim",
        type=str,
        default=None,
        help="Separator if you want to use several target columns"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _=None):
    if args.in_csv is not None:
        df = pd.read_csv(args.in_csv)
    elif args.in_dir is not None:
        df = prepare_df_from_dirs(args.in_dir, args.tag_column)
    else:
        raise Exception

    if args.tag_delim is not None:
        df = separate_tags(
            df, tag_column=args.tag_column, tag_delim=args.tag_delim
        )

    tag2lbl = prepare_dataset_labeling(df, args.tag_column)
    print("Num classes: ", len(tag2lbl))

    with open(args.out_labeling, "w") as fout:
        json.dump(tag2lbl, fout, indent=4)

    if args.out_dataset is not None:
        df.to_csv(args.out_dataset, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
