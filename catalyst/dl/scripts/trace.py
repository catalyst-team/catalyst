from typing import Dict, Union
import argparse
from argparse import ArgumentParser
from pathlib import Path

import torch

from catalyst.dl import Experiment, utils
from catalyst.utils.tools.typing import Device


def trace_model_from_checkpoint(
    logdir: Path,
    method_name: str,
    checkpoint_name: str,
    stage: str = None,
    loader: Union[str, int] = None,
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    device: Device = "cpu",
):
    """Traces model using created experiment and runner.

    Args:
        logdir (Union[str, Path]): Path to Catalyst logdir with model
        checkpoint_name (str): Name of model checkpoint to use
        stage (str): experiment's stage name
        loader (Union[str, int]): experiment's loader name or its index
        method_name (str): Model's method name that will be
            used as entrypoint during tracing
        mode (str): Mode for model to trace (``train`` or ``eval``)
        requires_grad (bool): Flag to use grads
        opt_level (str): AMP FP16 init level
        device (str): Torch device

    Returns:
        the traced model
    """
    config_path = logdir / "configs" / "_config.json"
    checkpoint_path = logdir / "checkpoints" / f"{checkpoint_name}.pth"
    print("Load config")
    config: Dict[str, dict] = utils.load_config(config_path)
    runner_params = config.get("runner_params", {}) or {}

    # Get expdir name
    config_expdir = Path(config["args"]["expdir"])
    # We will use copy of expdir from logs for reproducibility
    expdir = Path(logdir) / "code" / config_expdir.name

    print("Import experiment and runner from logdir")
    ExperimentType, RunnerType = utils.import_experiment_and_runner(expdir)
    experiment: Experiment = ExperimentType(config)

    print(f"Load model state from checkpoints/{checkpoint_name}.pth")
    if stage is None:
        stage = list(experiment.stages)[0]

    model = experiment.get_model(stage)
    checkpoint = utils.load_checkpoint(checkpoint_path)
    utils.unpack_checkpoint(checkpoint, model=model)

    runner: RunnerType = RunnerType(**runner_params)
    runner.model, runner.device = model, device

    if loader is None:
        loader = 0
    batch = experiment.get_native_batch(stage, loader)

    print("Tracing")
    traced = utils.trace_model(
        model=model,
        runner=runner,
        batch=batch,
        method_name=method_name,
        mode=mode,
        requires_grad=requires_grad,
        opt_level=opt_level,
        device=device,
    )

    print("Done")
    return traced


def build_args(parser: ArgumentParser):
    """Builds the command line parameters."""
    parser.add_argument("logdir", type=Path, help="Path to model logdir")
    parser.add_argument(
        "--method", "-m", default="forward", help="Model method to trace"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        default="best",
        help="Checkpoint's name to trace",
        metavar="CHECKPOINT_NAME",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory to save traced model",
    )
    parser.add_argument(
        "--out-model",
        type=Path,
        default=None,
        help="Output path to save traced model (overrides --out-dir)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "train"],
        default="eval",
        help="Model's mode 'eval' or 'train'",
    )
    parser.add_argument(
        "--with-grad",
        action="store_true",
        default=False,
        help="If true, model will be traced with `requires_grad_(True)`",
    )
    parser.add_argument(
        "--opt-level",
        type=str,
        default=None,
        help="Opt level for FP16 (optional)",
    )

    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Stage from experiment from which model and loader will be taken",
    )

    parser.add_argument(
        "--loader",
        type=str,
        default=None,
        help="Loader name to get the batch from",
    )

    return parser


def parse_args():
    """Parses the command line arguments for the main method."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


def main(args, _):
    """Main method for ``catalyst-dl trace``."""
    logdir: Path = args.logdir
    method_name: str = args.method
    checkpoint_name: str = args.checkpoint
    mode: str = args.mode
    requires_grad: bool = args.with_grad
    opt_level: str = args.opt_level

    if opt_level is not None:
        opt_level = opt_level
        device = "cuda"
    else:
        opt_level = None
        device = "cpu"

    traced = trace_model_from_checkpoint(
        logdir,
        method_name,
        checkpoint_name=checkpoint_name,
        stage=args.stage,
        loader=args.loader,
        mode=mode,
        requires_grad=requires_grad,
        opt_level=opt_level,
        device=device,
    )

    if args.out_model is None:
        file_name = utils.get_trace_name(
            method_name=method_name,
            mode=mode,
            requires_grad=requires_grad,
            opt_level=opt_level,
            additional_string=checkpoint_name,
        )

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
