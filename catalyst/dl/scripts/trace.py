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
    checkpoint_name: str,
    mode: str = "eval",
    requires_grad: bool = False,
):
    config_path = logdir / "configs" / "_config.json"
    checkpoint_path = logdir / "checkpoints" / f"{checkpoint_name}.pth"
    print("Load config")
    config: Dict[str, dict] = safitty.load(config_path)

    # Get expdir name
    config_expdir = safitty.get(config, "args", "expdir", apply=Path)
    # We will use copy of expdir from logs for reproducibility
    expdir = Path(logdir) / "code" / config_expdir.name

    print("Import experiment and runner from logdir")
    ExperimentType, RunnerType = import_experiment_and_runner(expdir)
    experiment: Experiment = ExperimentType(config)

    print(f"Load model state from checkpoints/{checkpoint_name}.pth")
    model = experiment.get_model(next(iter(experiment.stages)))
    checkpoint = utils.load_checkpoint(checkpoint_path)
    utils.unpack_checkpoint(checkpoint, model=model)

    print("Tracing")
    traced = trace_model(
        model, experiment, RunnerType,
        method_name=method_name,
        mode=mode,
        requires_grad=requires_grad,
    )

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
        help="Checkpoint's name to trace",
        metavar="CHECKPOINT_NAME"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory to save traced model"
    )
    parser.add_argument(
        "--out-model",
        type=Path,
        default=None,
        help="Output path to save traced model (overrides --out-dir)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "train"],
        default="eval",
        help="Model's mode 'eval' or 'train'"
    )
    parser.add_argument(
        "--with-grad",
        action="store_true",
        default=False,
        help="If true, model will be traced with `requires_grad_(True)`"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _):
    logdir: Path = args.logdir
    method_name: str = args.method
    checkpoint_name: str = args.checkpoint
    mode: str = args.mode
    requires_grad: bool = args.with_grad

    traced = trace_model_from_checkpoint(
        logdir, method_name,
        checkpoint_name=checkpoint_name,
        mode=mode,
        requires_grad=requires_grad,
    )

    if args.out_model is None:
        file_name = f"traced-{checkpoint_name}-{method_name}"
        if mode == "train":
            file_name += "-in_train"

        if requires_grad:
            file_name += f"-with_grad"
        file_name += ".pth"

        output: Path = args.out_dir
        if output is None:
            output: Path = logdir / "trace"
        output.mkdir(exist_ok=True, parents=True)

        out_model = str(output / file_name)
    else:
        out_model = str(args.out_model)

    torch.jit.save(traced, out_model)


if __name__ == "__main__":
    main(parse_args(), None)
