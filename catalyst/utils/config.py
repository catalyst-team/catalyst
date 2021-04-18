from typing import Dict, List, Union
from collections import OrderedDict
import json
from logging import getLogger
from pathlib import Path
import re

import yaml

logger = getLogger(__name__)


class OrderedLoader(yaml.SafeLoader):
    pass


def construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))


OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
OrderedLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _load_ordered_yaml(stream):
    return yaml.load(stream, OrderedLoader)


def load_config(
    path: Union[str, Path],
    ordered: bool = False,
    data_format: str = None,
    encoding: str = "utf-8",
) -> Union[Dict, List]:
    """
    Loads config by giving path. Supports YAML and JSON files.

    Examples:
        >>> load(path="./config.yml", ordered=True)

    Args:
        path: path to config file (YAML or JSON)
        ordered: if true the config will be loaded as ``OrderedDict``
        data_format: ``yaml``, ``yml`` or ``json``.
        encoding: encoding to read the config

    Returns:
        Union[Dict, List]: config

    Raises:
        ValueError: if path ``path`` doesn't exists
            or file format is not YAML or JSON

    Adapted from
    https://github.com/TezRomacH/safitty/blob/v1.2.0/safitty/parser.py#L63
    which was adapted from
    https://github.com/catalyst-team/catalyst/blob/v19.03/catalyst/utils/config.py#L10
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Path '{path}' doesn't exist!")

    if data_format is not None:
        suffix = data_format.lower()
        if not suffix.startswith("."):
            suffix = f".{suffix}"
    else:
        suffix = path.suffix

    assert suffix in [".json", ".yml", ".yaml"], f"Unknown file format '{suffix}'"

    config = None
    with path.open(encoding=encoding) as stream:
        if suffix == ".json":
            object_pairs_hook = OrderedDict if ordered else None
            file = "\n".join(stream.readlines())
            if file != "":
                config = json.loads(file, object_pairs_hook=object_pairs_hook)

        elif suffix in [".yml", ".yaml"]:
            loader = OrderedLoader if ordered else yaml.SafeLoader
            config = yaml.load(stream, loader)

    if config is None:
        return {}

    return config


def save_config(
    config: Union[Dict, List],
    path: Union[str, Path],
    data_format: str = None,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    """
    Saves config to file. Path must be either YAML or JSON.

    Args:
        config (Union[Dict, List]): config to save
        path (Union[str, Path]): path to save
        data_format: ``yaml``, ``yml`` or ``json``.
        encoding: Encoding to write file. Default is ``utf-8``
        ensure_ascii: Used for JSON, if True non-ASCII
        characters are escaped in JSON strings.
        indent: Used for JSON

    Adapted from
    https://github.com/TezRomacH/safitty/blob/v1.2.0/safitty/parser.py#L110
    which was adapted from
    https://github.com/catalyst-team/catalyst/blob/v19.03/catalyst/utils/config.py#L38
    """
    path = Path(path)

    if data_format is not None:
        suffix = data_format
    else:
        suffix = path.suffix

    assert suffix in [".json", ".yml", ".yaml"], f"Unknown file format '{suffix}'"

    with path.open(encoding=encoding, mode="w") as stream:
        if suffix == ".json":
            json.dump(config, stream, indent=indent, ensure_ascii=ensure_ascii)
        elif suffix in [".yml", ".yaml"]:
            yaml.dump(config, stream)


__all__ = ["load_config", "save_config"]
