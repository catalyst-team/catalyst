# flake8: noqa
# @TODO: code formatting issue for 20.07 release
import argparse
from pathlib import Path

from catalyst.dl import utils


def build_args(parser):
    """Constructs the command-line arguments for ``catalyst-dl init``."""
    parser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        default=None,
        choices=["empty", "classification", "segmentation", "detection"],
        help="select a Catalyst pipeline",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="use interactive wizard to setup Catalyst pipeline",
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default="./", help="path where to init"
    )

    return parser


def parse_args():
    """Parses the command line arguments for the main method."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _):
    """Run the ``catalyst-dl init`` script."""
    if args.interactive:
        utils.run_wizard()
    else:
        utils.clone_pipeline(args.pipeline, args.out_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args, None)
