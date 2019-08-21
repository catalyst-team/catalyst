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


def trace_model_from_checkpoint(
    logdir: Path,
    method_name: str,
    checkpoint_name: str
):
    config_path = logdir / "configs/_config.json"
    checkpoint_path = logdir / f"checkpoints/{checkpoint_name}.pth"
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

    print(f"Load model state from checkpoints/{checkpoint_name}.pth")
    model = experiment.get_model(next(iter(experiment.stages)))
    checkpoint = utils.load_checkpoint(checkpoint_path)
    utils.unpack_checkpoint(checkpoint, model=model)

    print("Tracing")
    traced = trace_model(model, experiment, RunnerType, method_name)

    print("Done")
    return traced


def build_args(parser: ArgumentParser):
    parser.add_argument(
        "logdir",
        type=Path,
        help="Path to model logdir"
    )
    parser.add_argument(
        "--method", "-m",
        default="forward",
        help="Model method to trace"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="best",
        help="Checkpoint's name to trace"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory to save traced model"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _):
    logdir = args.logdir
    method_name = args.method
    checkpoint_name = args.checkpoint

    traced = trace_model_from_checkpoint(logdir, method_name, checkpoint_name)
    output: Path = args.out_dir
    if output is None:
        output: Path = logdir / "trace"
    output.mkdir(exist_ok=True, parents=True)
    torch.jit.save(traced, str(output / f"traced-{checkpoint_name}.pth"))


if __name__ == "__main__":
    main(parse_args(), None)
