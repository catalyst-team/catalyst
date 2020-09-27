#!/usr/bin/env python

import argparse
from argparse import ArgumentParser
import logging
from pathlib import Path

from catalyst.dl.utils.swa import generate_averaged_weights


def build_args(parser: ArgumentParser):
    """Builds the command line parameters."""
    parser.add_argument(
        "logdir", 
        type=Path, 
        help="Path to models logdir"
    )
    parser.add_argument(
        "--models_mask", 
        "-m", 
        type=str,
        help="Pattern for models to average"
    )
    parser.add_argument(
        "--save-avaraged-model",
        "-s",
        type=bool,
        default=True,
        help="Flag for saving avaraged model",
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
    save_avaraged_model: bool = args.save_avaraged_model

    averaged_weights = generate_averaged_weights(
        logdir,
        models_mask,
        save_avaraged_model=save_avaraged_model
    )

if __name__ == "__main__":
    main(parse_args(), None)
