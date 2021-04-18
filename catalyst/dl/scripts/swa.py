import argparse
from argparse import ArgumentParser
from pathlib import Path

import torch

from catalyst.utils.swa import get_averaged_weights_by_path_mask


def build_args(parser: ArgumentParser):
    """Builds the command line parameters."""
    parser.add_argument(
        "--model-mask",
        "--models-mask",
        "-m",
        type=str,
        default="*.pth",
        help="Pattern for models to average",
        dest="models_mask",
    )
    parser.add_argument("--logdir", type=Path, default=None, help="Path to experiment logdir")
    parser.add_argument(
        "--output-path", type=Path, default="./swa.pth", help="Path to save averaged model",
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

    averaged_weights = get_averaged_weights_by_path_mask(path_mask=models_mask, logdir=logdir)

    torch.save(averaged_weights, str(output_path))


if __name__ == "__main__":
    main(parse_args(), None)
