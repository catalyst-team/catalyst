from typing import TYPE_CHECKING, Dict, Union  # isort:skip
from pathlib import Path

from catalyst.dl import Experiment
from catalyst.dl.utils import trace_model
from catalyst.utils.tools.typing import Device
from catalyst.utils import (
    import_experiment_and_runner,
    unpack_checkpoint,
    get_native_batch_from_loaders,
    load_checkpoint,
    load_config,
)

if TYPE_CHECKING:
    from catalyst.dl import Runner  # noqa: F401


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
    """
    Traces model using created experiment and runner.

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
    config: Dict[str, dict] = load_config(config_path)
    runner_params = config.get("runner_params", {}) or {}

    # Get expdir name
    config_expdir = Path(config["args"]["expdir"])
    # We will use copy of expdir from logs for reproducibility
    expdir = Path(logdir) / "code" / config_expdir.name

    print("Import experiment and runner from logdir")
    ExperimentType, RunnerType = import_experiment_and_runner(expdir)
    experiment: Experiment = ExperimentType(config)

    print(f"Load model state from checkpoints/{checkpoint_name}.pth")
    if stage is None:
        stage = list(experiment.stages)[0]

    model = experiment.get_model(stage)
    checkpoint = load_checkpoint(checkpoint_path)
    unpack_checkpoint(checkpoint, model=model)

    runner: RunnerType = RunnerType(**runner_params)
    runner.model, runner.device = model, device

    if loader is None:
        loader = 0
    batch = get_native_batch_from_loaders(experiment.get_loaders(stage), loader)

    # function to run prediction on batch
    def predict_fn(model, inputs, **kwargs):
        _model = runner.model
        runner.model = model
        result = runner.predict_batch(inputs, **kwargs)
        runner.model = _model
        return result

    print("Tracing")
    traced_model = trace_model(
        model=model,
        predict_fn=predict_fn,
        batch=batch,
        method_name=method_name,
        mode=mode,
        requires_grad=requires_grad,
        opt_level=opt_level,
        device=device,
    )

    print("Done")
    return traced_model


__all__ = [
    "trace_model_from_checkpoint"
]
