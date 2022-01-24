#!/usr/bin/env python
from typing import Iterable, List
import argparse
from argparse import ArgumentParser
from pathlib import Path

import torch

from catalyst.dl.scripts._misc import get_model_from_logdir
from catalyst.utils.onnx import onnx_export


def build_args(parser: ArgumentParser):
    """Constructs the command-line arguments for ``catalyst-dl onnx``."""
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--logdir", type=Path, default=None, help="Path to logs directory")
    parser.add_argument("--shape", nargs="+", type=int, required=True, help="Shape of the input")
    parser.add_argument(
        "--method", type=str, default="forward", help="Forward pass method to be converted"
    )
    parser.add_argument(
        "--input-names", type=Iterable, default=None, help="Name of inputs in graph"
    )
    parser.add_argument(
        "--output-names", type=Iterable, default=None, help="Name of outputs in graph"
    )
    parser.add_argument("--dynamic-axes", type=dict, default=None, help="Axes with dynamic shapes")
    parser.add_argument("--opset-version", type=int, default=9, help="Version of onnx opset")
    parser.add_argument(
        "--do-constant-folding",
        type=bool,
        default=False,
        help="If True, the constant-folding optimization is applied to the model during export",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="If specified, we will print out a debug description of the trace being exported.",
    )
    parser.add_argument(
        "--output-path", type=Path, default="./onnx_model.pth", help="Path to save onnx model"
    )

    return parser


def parse_args():
    """Parses the command line arguments and returns arguments and config."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, _):
    """Runs the ``catalyst-dl onnx`` script."""
    checkpoint_path: Path = args.checkpoint_path
    logdir: Path = args.logdir
    shape: List[int] = args.shape
    method: str = args.method
    input_names: Iterable = args.input_names
    output_names: List[str] = args.output_names
    dynamic_axes: dict = args.dynamic_axes
    opset_version: int = args.opset_version
    do_constant_folding: bool = args.do_constant_folding
    verbose: bool = args.verbose
    output_path: Path = args.output_path

    model = get_model_from_logdir(checkpoint_path, logdir)

    batch = torch.rand(shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_export(
        model=model,
        file=output_path,
        batch=batch,
        method_name=method,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        verbose=verbose,
    )


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
