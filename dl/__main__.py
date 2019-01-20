from argparse import ArgumentParser

from .scripts import train, inference

COMMANDS = {
    'train': train.run_train,
    'infer': inference.run_inference,
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser('catalyst-dl')

    subparsers = parser.add_subparsers(
        metavar='{command}',
        dest='command',
        help='train or infer',
    )
    subparsers.required = True

    train.build_args(subparsers.add_parser('train'))
    inference.build_args(subparsers.add_parser('infer'))

    return parser


def main():
    parser = build_parser()

    args, uargs = parser.parse_known_args()

    COMMANDS[args.command](args, uargs)


if __name__ == '__main__':
    main()
