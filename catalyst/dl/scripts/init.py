import argparse
from pathlib import Path

from catalyst.dl import utils


def build_args(parser):
    parser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        default=None,
        choices=["empty", "classification", "segmentation", "detection"],
        help="select a Catalyst pipeline"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="use interactive wizard to setup Catalyst pipeline"
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default="./", help="path where to init"
    )

    return parser


def main(args, _):
    if args.interactive:
        utils.run_wizard()
    else:
        utils.clone_pipeline(args.pipeline, args.out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args, None)
