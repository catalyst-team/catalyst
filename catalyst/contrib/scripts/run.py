#!/usr/bin/env python
from typing import Dict, Iterable
import argparse
import logging

from catalyst import utils
from catalyst.registry import REGISTRY


def parse_args(args: Iterable = None, namespace: argparse.Namespace = None):
    """Parses the command line arguments and returns arguments and config."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "--configs",
        "-C",
        nargs="+",
        default=("config.yml",),
        type=str,
        help="path to config/configs",
        metavar="CONFIG_PATH",
        dest="configs",
    )

    utils.boolean_flag(
        parser,
        "deterministic",
        default=None,
        help="Deterministic mode if running in CuDNN backend",
    )
    utils.boolean_flag(parser, "benchmark", default=None, help="Use CuDNN benchmark")

    args, unknown_args = parser.parse_known_args(args=args, namespace=namespace)
    return vars(args), unknown_args


def process_configs(
    configs: Iterable[str], deterministic: bool = None, benchmark: bool = None
) -> Dict:
    """Merges YAML configs and prepares env."""
    # there is no way to set deterministic/benchmark flags with a runner,
    # so do it manually
    utils.prepare_cudnn(deterministic, benchmark)

    config = {}
    for config_path in configs:
        config_part = utils.load_config(config_path, ordered=True)
        config = utils.merge_dicts(config, config_part)

    return config


def run_from_params(experiment_params: Dict) -> None:
    """Runs multi-stage experiment."""
    logger = logging.getLogger(__name__)

    runner = experiment_params["runner"]
    for stage_params in experiment_params["run"]:
        name = stage_params.pop("_call_")
        func = getattr(runner, name)

        result = func(**stage_params)
        if result is not None:
            logger.info(f"{name}:\n{result}")


def run_from_config(
    configs: Iterable[str],
    deterministic: bool = None,
    benchmark: bool = None,
) -> None:
    """Creates Runner from YAML configs and runs experiment."""
    config = process_configs(configs, deterministic=deterministic, benchmark=benchmark)
    experiment_params = REGISTRY.get_from_params(**config)
    run_from_params(experiment_params)


def main():
    """Runs the ``catalyst-run`` script."""
    kwargs, _ = parse_args()
    run_from_config(**kwargs)


if __name__ == "__main__":
    main()
