#!/usr/bin/env python
import argparse
from argparse import ArgumentParser
from pathlib import Path

import torch

from catalyst.utils import get_averaged_weights_by_path_mask


def build_args(parser: ArgumentParser):
    """Constructs the command-line arguments for ``catalyst-dl swa``."""
    parser.add_argument(
        "--path-mask",
        type=str,
        default="*.pth",
        help="globe-like pattern for models to average",
    )
    parser.add_argument("--logdir", type=Path, default=None, help="Path to logs directory")
    parser.add_argument(
        "--output-path", type=Path, default="./swa.pth", help="Path to save averaged model"
    )

    return parser


def parse_args():
    """Parses the command line arguments and returns arguments and config."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, _):
    """Runs the ``catalyst-dl swa`` script."""
    logdir: Path = args.logdir
    path_mask: str = args.path_mask
    output_path: Path = args.output_path

    averaged_weights = get_averaged_weights_by_path_mask(path_mask=path_mask, logdir=logdir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged_weights, output_path.as_posix())


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
