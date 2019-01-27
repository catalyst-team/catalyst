# -*- coding: utf-8 -*-
"""Catalyst-data scripts.


1. **tag2label** prepares a dataset to json like `{"class_id":  class_column_from_dataset}`
2. **check-images** checks images in your data to be non-broken and writes a flag:
true if image opened without an error and false otherwise


Examples:

    .. code:: bash

        catalyst-data tag2label \\
            --in-dir=./data/ants_bees \\
            --out-dataset=./data/ants_bees/dataset.csv \\
            --out-labeling=./data/ants_bees/tag2cls.json

    .. code:: bash

        catalyst-data check-images \\
            --in-csv=./data/input.csv \\
            --img-datapath=./data/images
            --img-col="filename" \\
            --out-csv=./data/input_checked.csv \\
            --n-cpu=4

"""

from collections import OrderedDict
from argparse import ArgumentParser, RawTextHelpFormatter

from catalyst.contrib.scripts import check_images, \
    image2embedding, tag2label

COMMANDS = OrderedDict(
    [
        ("tag2label", tag2label),
        ("check-images", check_images),
        ("image2embedding", image2embedding),
    ]
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        "catalyst-data", formatter_class=RawTextHelpFormatter
    )
    all_commands = ', \n'.join(map(lambda x: f"    {x}", COMMANDS.keys()))

    subparsers = parser.add_subparsers(
        metavar="{command}",
        dest="command",
        help=f"available commands: \n{all_commands}",
    )
    subparsers.required = True

    for key, value in COMMANDS.items():
        value.build_args(subparsers.add_parser(key))

    return parser


def main():
    parser = build_parser()

    args, uargs = parser.parse_known_args()

    COMMANDS[args.command].main(args, uargs)


if __name__ == "__main__":
    main()
