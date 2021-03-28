from argparse import ArgumentParser, RawTextHelpFormatter
from collections import OrderedDict

from catalyst.__version__ import __version__
from catalyst.dl.scripts import run, swa  # , trace
from catalyst.settings import SETTINGS

COMMANDS = OrderedDict([("run", run), ("swa", swa)])  # ("trace", trace)


# if SETTINGS.use_quantization:
#     from catalyst.dl.scripts import quantize
#
#     COMMANDS["quantize"] = quantize

if SETTINGS.optuna_required:

    from catalyst.dl.scripts import tune

    COMMANDS["tune"] = tune

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
