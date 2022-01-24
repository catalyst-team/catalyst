#!/usr/bin/env python
from typing import List, Optional, Union
import argparse
from argparse import ArgumentParser
from pathlib import Path

from catalyst.dl.scripts._misc import get_model_from_logdir
from catalyst.utils.pruning import prune_model
from catalyst.utils.torch import pack_checkpoint, save_checkpoint


def build_args(parser: ArgumentParser):
    """Constructs the command-line arguments for ``catalyst-dl quantize``."""
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--logdir", type=Path, required=True, help="Path to logs directory")
    parser.add_argument("--pruning_fn", type=str, required=True, help="Pruning function name")
    parser.add_argument(
        "--amount", type=Union[float, int], required=True, help="Quantity of parameters to prune"
    )
    parser.add_argument(
        "--keys_to_prune", type=List[str], default=None, help="List of tensors in modules to prune"
    )
    parser.add_argument(
        "--loglayers_to_prunedir", type=List[str], default=None, help="Module names to be pruned"
    )
    parser.add_argument(
        "--dim", type=int, default=None, help="Dimension for structured pruning methods"
    )
    parser.add_argument(
        "--l_norm", type=int, default=None, help="L-norm in case of using ln_structured"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default="./pruned_model.pth",
        help="Path to save quantized model",
    )

    return parser


def parse_args():
    """Parses the command line arguments and returns arguments and config."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, _):
    """Runs the ``catalyst-dl prune`` script."""
    checkpoint_path: Path = args.checkpoint_path
    logdir: Path = args.logdir
    pruning_fn: str = args.pruning_fn
    amount: Union[float, int] = args.dtype
    keys_to_prune: Optional[List[str]] = args.keys_to_prune
    layers_to_prune: Optional[List[str]] = args.layers_to_prune
    dim: int = args.dim
    l_norm: int = args.l_norm
    output_path: Path = args.output_path

    model = get_model_from_logdir(checkpoint_path, logdir)

    prune_model(model, pruning_fn, amount, keys_to_prune, layers_to_prune, dim, l_norm)
    checkpoint = pack_checkpoint(model=model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint=checkpoint, path=output_path)


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
