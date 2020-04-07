from typing import Optional
import argparse


def boolean_flag(
    parser: argparse.ArgumentParser,
    name: str,
    default: Optional[bool] = False,
    help: str = None,
    shorthand: str = None,
) -> None:
    """Add a boolean flag to a parser inplace.

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> boolean_flag(
        >>>     parser, "flag", default=False, help="some flag", shorthand="f"
        >>> )

    Args:
        parser (argparse.ArgumentParser): parser to add the flag to
        name (str): argument name
            --<name> will enable the flag,
            while --no-<name> will disable it
        default (bool, optional): default value of the flag
        help (str): help string for the flag
        shorthand (str): shorthand string for the argument
    """
    dest = name.replace("-", "_")
    names = ["--" + name]
    if shorthand is not None:
        names.append("-" + shorthand)
    parser.add_argument(
        *names, action="store_true", default=default, dest=dest, help=help
    )
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


__all__ = ["boolean_flag"]
