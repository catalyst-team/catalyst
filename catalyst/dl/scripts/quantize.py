#!/usr/bin/env python
import argparse
from argparse import ArgumentParser
from pathlib import Path

from catalyst.dl.scripts._misc import get_model_from_logdir
from catalyst.utils.quantization import quantize_model
from catalyst.utils.torch import pack_checkpoint, save_checkpoint


def build_args(parser: ArgumentParser):
    """Constructs the command-line arguments for ``catalyst-dl quantize``."""
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--logdir", type=Path, default=None, help="Path to logs directory")
    parser.add_argument(
        "--qconfig_spec", type=dict, default=None, help="Quantization config in PyTorch format"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="qint8",
        help="Type of weights after quantization",
    )
    parser.add_argument(
        "--output-path", type=Path, default="./q_model.pth", help="Path to save quantized model"
    )

    return parser


def parse_args():
    """Parses the command line arguments and returns arguments and config."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, _):
    """Runs the ``catalyst-dl quantize`` script."""
    checkpoint_path: Path = args.checkpoint_path
    logdir: Path = args.logdir
    qconfig_spec: dict = args.qconfig_spec
    dtype: str = args.dtype
    output_path: Path = args.output_path

    model = get_model_from_logdir(checkpoint_path, logdir)

    q_model = quantize_model(model.cpu(), qconfig_spec=qconfig_spec, dtype=dtype)
    checkpoint = pack_checkpoint(model=q_model)
    save_checkpoint(checkpoint=checkpoint, path=output_path)


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
