import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import safitty
import torch

from catalyst.dl.utils.scripts import import_experiment_and_runner
from catalyst.dl.core import Experiment
from catalyst import utils
from catalyst.dl.utils.trace import trace_model


def trace_model_from_checkpoint(logdir, method_name):
    config_path = logdir / "configs/_config.json"
    checkpoint_path = logdir / "checkpoints/best.pth"
    print("Load config")
    config: Dict[str, dict] = safitty.load(config_path)

    # Get expdir name
    config_expdir = Path(config["args"]["expdir"])
    # We will use copy of expdir from logs for reproducibility
    expdir_from_logs = Path(logdir) / "code" / config_expdir.name

    print("Import experiment and runner from logdir")
    ExperimentType, RunnerType = \
        import_experiment_and_runner(expdir_from_logs)
    experiment: Experiment = ExperimentType(config)

    print("Load model state from checkpoints/best.pth")
    model = experiment.get_model(next(iter(experiment.stages)))
    checkpoint = utils.load_checkpoint(checkpoint_path)
    utils.unpack_checkpoint(checkpoint, model=model)

    print("Tracing")
    traced = trace_model(model, experiment, RunnerType, method_name)

    print("Done")
    return traced


def build_args(parser: ArgumentParser):
    parser.add_argument("logdir", type=Path)
    parser.add_argument(
        "--method", "-m",
        default="forward",
        help="Model method to trace")

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _):
    logdir = args.logdir
    method_name = args.method

    traced = trace_model_from_checkpoint(logdir, method_name)
    torch.jit.save(traced, str(logdir / "traced.pth"))


if __name__ == "__main__":
    main(parse_args(), None)
