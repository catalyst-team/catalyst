from collections import OrderedDict
from argparse import ArgumentParser, RawTextHelpFormatter

from catalyst.contrib.scripts import project_embeddings, \
    check_index_model, create_index_model

COMMANDS = OrderedDict(
    [
        ("check-index-model", check_index_model),
        ("create-index-model", create_index_model),
        ("project-embeddings", project_embeddings)
    ]
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        "catalyst-contrib", formatter_class=RawTextHelpFormatter
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
