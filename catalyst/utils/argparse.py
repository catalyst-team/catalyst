import argparse
from typing import Any, Optional


def args_are_not_none(*args: Optional[Any]) -> bool:
    """
    Check that all arguments are not None
    Args:
        *args (Any): values
    Returns:
         bool: True if all value were not None, False otherwise
    """
    result = args is not None
    if not result:
        return result

    for arg in args:
        if arg is None:
            result = False
            break

    return result


def boolean_flag(
    parser: argparse.ArgumentParser,
    name: str,
    default: Optional[bool] = False,
    help: str = None,
    shorthand: str = None,
) -> None:
    """
    Add a boolean flag to a parser inplace.

    Args:
        parser (argparse.ArgumentParser): parser to add the flag to
        name (str): argument name
            --<name> will enable the flag,
            while --no-<name> will disable it
        default (bool, optional): default value of the flag
        help (str): help string for the flag
        shorthand (str): shorthand string for the argument

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> boolean_flag(
        >>>     parser, "flag", default=False, help="some flag", shorthand="f"
        >>> )
    """
    dest = name.replace("-", "_")
    names = ["--" + name]
    if shorthand is not None:
        names.append("-" + shorthand)
    parser.add_argument(
        *names, action="store_true", default=default, dest=dest, help=help
    )
    parser.add_argument("--no-" + name, action="store_false", dest=dest)
