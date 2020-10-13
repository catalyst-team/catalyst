import argparse
from argparse import ArgumentParser
from pathlib import Path

import torch

from catalyst.dl.utils.swa import generate_averaged_weights


def build_args(parser: ArgumentParser):
    """Builds the command line parameters."""
    parser.add_argument(
        "--logdir", type=Path, default=None, help="Path to models logdir"
    )
    parser.add_argument(
        "--models-mask",
        "-m",
        type=str,
        default="*.pth",
        help="Pattern for models to average",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default="./swa.pth",
        help="Path to save averaged model",
    )

    return parser


def parse_args():
    """Parses the command line arguments for the main method."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _):
    """Main method for ``catalyst-dl swa``."""
    logdir: Path = args.logdir
    models_mask: str = args.models_mask
    output_path: Path = args.output_path

    averaged_weights = generate_averaged_weights(logdir, models_mask)

    torch.save(averaged_weights, str(output_path))


if __name__ == "__main__":
    main(parse_args(), None)
