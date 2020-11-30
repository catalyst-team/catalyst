from argparse import ArgumentParser, RawTextHelpFormatter
from collections import OrderedDict
import logging

from catalyst.__version__ import __version__
from catalyst.dl.scripts import quantize, run, swa, trace
from catalyst.settings import SETTINGS

logger = logging.getLogger(__name__)

COMMANDS = OrderedDict(
    [("quantize", quantize), ("run", run), ("swa", swa), ("trace", trace)]
)

if SETTINGS.IS_OPTUNA_AVAILABLE:
    from catalyst.dl.scripts import tune

    COMMANDS["tune"] = tune

try:
    from catalyst.dl.scripts import init

    COMMANDS["init"] = init
except ImportError as ex:
    if SETTINGS.ipython_required:
        logger.warning(
            "ipython not available, to install ipython,"
            " run `pip install ipython`."
        )
        raise ex


COMMANDS = OrderedDict(sorted(COMMANDS.items()))


def build_parser() -> ArgumentParser:
    """Builds parser.

    Returns:
        parser
    """
    parser = ArgumentParser(
        "catalyst-dl", formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    all_commands = ", \n".join(map(lambda x: f"    {x}", COMMANDS.keys()))

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
    """catalyst-dl entry point."""
    parser = build_parser()

    args, uargs = parser.parse_known_args()

    COMMANDS[args.command].main(args, uargs)


if __name__ == "__main__":
    main()
