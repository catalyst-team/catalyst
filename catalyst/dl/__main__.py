from argparse import ArgumentParser, RawTextHelpFormatter
from collections import OrderedDict
import logging

from catalyst.__version__ import __version__
from catalyst.dl.scripts import run, swa, trace
from catalyst.settings import SETTINGS

logger = logging.getLogger(__name__)

COMMANDS = OrderedDict([("run", run), ("swa", swa), ("trace", trace)])


if SETTINGS.IS_QUANTIZATION_AVAILABLE:
    from catalyst.dl.scripts import quantize

    COMMANDS["quantize"] = quantize

try:
    import optuna  # noqa: F401

    from catalyst.dl.scripts import tune

    COMMANDS["tune"] = tune
except ImportError as ex:
    if SETTINGS.optuna_required:
        logger.warning(
            "catalyst[tune] requirements are not available, to install them,"
            " run `pip install catalyst[tune]`."
        )
        raise ex

try:
    from git import Repo as repo  # noqa: N813 F401
    from prompt_toolkit import prompt  # noqa: F401

    from catalyst.dl.scripts import init

    COMMANDS["init"] = init
except ImportError as ex:
    if SETTINGS.ml_required:
        logger.warning(
            "catalyst[ml] requirements are not available, to install them,"
            " run `pip install catalyst[ml]`."
        )
        raise ex


COMMANDS = OrderedDict(sorted(COMMANDS.items()))


def build_parser() -> ArgumentParser:
    """Builds parser.

    Returns:
        parser
    """
    parser = ArgumentParser("catalyst-dl", formatter_class=RawTextHelpFormatter)
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
    """catalyst-dl entry point."""
    parser = build_parser()

    args, uargs = parser.parse_known_args()

    COMMANDS[args.command].main(args, uargs)


if __name__ == "__main__":
    main()
