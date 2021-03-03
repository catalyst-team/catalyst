# -*- coding: utf-8 -*-
r"""Catalyst-contrib scripts.

Examples:
    1.  **collect-env** outputs relevant system environment info.
    Diagnose your system and show basic information.
    Used to get detail info for better bug reporting.

    .. code:: bash

        $ catalyst-contrib collect-env

    2.  **process-images** reads raw data and outputs
    preprocessed resized images

    .. code:: bash

        $ catalyst-contrib process-images \\
            --in-dir /path/to/raw/data/ \\
            --out-dir=./data/dataset \\
            --num-workers=6 \\
            --max-size=224 \\
            --extension=png \\
            --clear-exif \\
            --grayscale \\
            --expand-dims

    3. **tag2label** prepares a dataset to json like
    `{"class_id":  class_column_from_dataset}`

    .. code:: bash

        $ catalyst-contrib tag2label \\
            --in-dir=./data/dataset \\
            --out-dataset=./data/dataset_raw.csv \\
            --out-labeling=./data/tag2cls.json

    4. **split-dataframe** split your dataset into train/valid folds

    .. code:: bash

        $ catalyst-contrib split-dataframe \\
            --in-csv=./data/dataset_raw.csv \\
            --tag2class=./data/tag2cls.json \\
            --tag-column=tag \\
            --class-column=class \\
            --n-folds=5 \\
            --train-folds=0,1,2,3 \\
            --out-csv=./data/dataset.csv
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
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    all_commands = ", \n".join(map(lambda x: f"    {x}", COMMANDS.keys()))

    subparsers = parser.add_subparsers(
        metavar="{command}", dest="command", help=f"available commands: \n{all_commands}",
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
