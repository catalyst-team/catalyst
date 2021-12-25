#!/usr/bin/env python
from typing import Sequence
import argparse
from argparse import ArgumentParser
from pathlib import Path

import torch

from catalyst.dl.scripts._misc import get_model_from_logdir
from catalyst.utils.tracing import trace_model


def build_args(parser: ArgumentParser):
    """Constructs the command-line arguments for ``catalyst-dl run``."""
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--logdir", type=Path, default=None, help="Path to logs directory")
    parser.add_argument(
        "--shape", nargs="+", type=int, required=True, help="Shape of the input example"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="forward",
        help="Model's method name that will be used as entrypoint during tracing",
    )
    parser.add_argument(
        "--output-path", type=Path, default="./traced_model.pth", help="Path to save traced model"
    )

    return parser


def parse_args():
    """Parses the command line arguments and returns arguments and config."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, _):
    """Runs the ``catalyst-dl trace`` script."""
    checkpoint_path: Path = args.checkpoint_path
    logdir: Path = args.logdir
    shape: Sequence[int] = args.shape
    method: str = args.method
    output_path: Path = args.output_path

    model = get_model_from_logdir(checkpoint_path, logdir)

    batch = torch.rand(shape)
    traced_model = trace_model(model=model, batch=batch, method_name=method)
    torch.jit.save(traced_model, output_path)


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
