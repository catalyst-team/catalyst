from typing import Iterable
import argparse
import logging

from catalyst import utils
from catalyst.registry import REGISTRY


def parse_args():
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

    args, unknown_args = parser.parse_known_args()
    return vars(args), unknown_args


def run_from_config(
    configs: Iterable[str],
    deterministic: bool = None,
    benchmark: bool = None,
) -> None:
    """Creates Runner from YAML configs and runs experiment."""
    logger = logging.getLogger(__name__)

    # there is no way to set deterministic/benchmark flags with a runner,
    # so do it manually
    utils.prepare_cudnn(deterministic, benchmark)

    config = {}
    for config_path in configs:
        config_part = utils.load_config(config_path, ordered=True)
        config = utils.merge_dicts(config, config_part)
    # config_copy = copy.deepcopy(config)

    experiment_params = REGISTRY.get_from_params(**config)

    runner = experiment_params["runner"]
    for stage_params in experiment_params["run"]:
        name = stage_params.pop("_call_")
        func = getattr(runner, name)

        result = func(**stage_params)
        if result is not None:
            logger.info(f"{name}:\n{result}")

    # TODO: check if needed
    # logdir = getattr(runner, "logdir", getattr(runner, "_logdir"), None)
    # if logdir and utils.get_rank() <= 0:
    #     utils.dump_environment(logdir=logdir, config=config_copy, configs_path=configs)


def main():
    """Runs the ``catalyst-run`` script."""
    kwargs, unknown_args = parse_args()
    run_from_config(**kwargs)


if __name__ == "__main__":
    main()
