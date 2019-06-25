from argparse import ArgumentParser, RawTextHelpFormatter
from collections import OrderedDict

from catalyst.__version__ import __version__
from .scripts import trace, run, make_report

COMMANDS = OrderedDict(
    [
        ("run", run),
        ("trace", trace),
        ("make-report", make_report),
    ]
)


def build_parser() -> ArgumentParser:
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
    parser = build_parser()

    args, uargs = parser.parse_known_args()

    COMMANDS[args.command].main(args, uargs)


if __name__ == "__main__":
    main()
