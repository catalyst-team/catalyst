# -*- coding: utf-8 -*-
r"""Catalyst-contrib scripts.

Examples:
    1.  **collect-env** outputs relevant system environment info.
    Diagnose your system and show basic information.
    Used to get detail info for better bug reporting.

    .. code:: bash

        $ catalyst-contrib collect-env

    2.  **project-embeddings**

    .. code:: bash

        $ catalyst-contrib process-images \\
            --in-npy="./embeddings.npy" \
            --in-csv="./images.csv" \
            --out-dir="." \
            --img-size=64 \
            --img-col="images" \
            --meta-cols="meta" \
            --img-rootpath="."
"""

from argparse import ArgumentParser, RawTextHelpFormatter
from collections import OrderedDict

from catalyst.__version__ import __version__
from catalyst.contrib.scripts import collect_env
from catalyst.settings import SETTINGS

COMMANDS = OrderedDict([("collect-env", collect_env)])

if SETTINGS.ml_required:
    from catalyst.contrib.scripts import project_embeddings, split_dataframe, tag2label

    COMMANDS["project-embeddings"] = project_embeddings
    COMMANDS["tag2label"] = tag2label
    COMMANDS["split-dataframe"] = split_dataframe


if SETTINGS.cv_required and SETTINGS.ml_required:
    from catalyst.contrib.scripts import process_images  # , image2embedding

    COMMANDS["process-images"] = process_images
    # COMMANDS["image2embedding"] = image2embedding

COMMANDS = OrderedDict(sorted(COMMANDS.items()))


def build_parser() -> ArgumentParser:
    """Builds parser.

    Returns:
        parser
    """
    parser = ArgumentParser("catalyst-contrib", formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    all_commands = ", \n".join(map(lambda x: f"    {x}", COMMANDS.keys()))

    subparsers = parser.add_subparsers(
        metavar="{command}", dest="command", help=f"available commands: \n{all_commands}"
    )
    subparsers.required = True

    for key, value in COMMANDS.items():
        value.build_args(subparsers.add_parser(key))

    return parser


def main():
    """catalyst-contrib entry point."""
    parser = build_parser()
    args, uargs = parser.parse_known_args()
    COMMANDS[args.command].main(args, uargs)


if __name__ == "__main__":
    main()
