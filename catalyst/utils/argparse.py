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
    parser, name: str, default: bool = False, help: str = None
) -> None:
    """
    Add a boolean flag to argparse parser.

    Args:
        parser (argparse.Parser): parser to add the flag to
        name (str): --<name> will enable the flag,
            while --no-<name> will disable it
        default (bool, optional): default value of the flag
        help (str): help string for the flag
    """
    dest = name.replace("-", "_")
    parser.add_argument(
        "--" + name,
        action="store_true",
        default=default,
        dest=dest,
        help=help
    )
    parser.add_argument("--no-" + name, action="store_false", dest=dest)
