from typing import Dict, Optional, Set, TYPE_CHECKING, Union
import logging
from pathlib import Path

import torch
from torch import quantization
from torch.nn import Module

from catalyst.tools import Model
from catalyst.utils import (
    import_experiment_and_runner,
    load_checkpoint,
    load_config,
    unpack_checkpoint,
)

if TYPE_CHECKING:
    from catalyst.dl.experiment.config import ConfigExperiment


logger = logging.getLogger(__name__)


def save_quantized_model(
    model: Module,
    logdir: Union[str, Path] = None,
    checkpoint_name: str = None,
    out_dir: Union[str, Path] = None,
    out_model: Union[str, Path] = None,
) -> None:
    """Saves quantized model.

    Args:
        model (ScriptModule): Traced model
        logdir (Union[str, Path]): Path to experiment
        checkpoint_name (str): name for the checkpoint
        out_dir (Union[str, Path]): Directory to save model to
            (overrides logdir)
        out_model (Union[str, Path]): Path to save model to
            (overrides logdir & out_dir)

    Raises:
        ValueError: if nothing out of `logdir`, `out_dir` or `out_model`
          is specified.
    """
    if out_model is None:
        file_name = f"{checkpoint_name}_quantized.pth"

        output: Path = out_dir
        if output is None:
            if logdir is None:
                raise ValueError(
                    "One of `logdir`, `out_dir` or `out_model` "
                    "should be specified"
                )
            output: Path = Path(logdir) / "quantized"

        output.mkdir(exist_ok=True, parents=True)

        out_model = str(output / file_name)
    else:
        out_model = str(out_model)

    torch.save(model.state_dict(), out_model)


def quantize_model_from_checkpoint(
    logdir: Path,
    checkpoint_name: str,
    stage: str = None,
    qconfig_spec: Optional[Union[Set, Dict]] = None,
    dtype: Optional[torch.dtype] = torch.qint8,
    backend: str = None,
) -> Model:
    """
    Quantize model using created experiment and runner.

    Args:
        logdir (Union[str, Path]): Path to Catalyst logdir with model
        checkpoint_name (str): Name of model checkpoint to use
        stage (str): experiment's stage name
        qconfig_spec: torch.quantization.quantize_dynamic
                parameter, you can define layers to be quantize
        dtype: type of the model parameters, default int8
        backend: defines backend for quantization

    Returns:
        Quantized model
    """
    if backend is not None:
        torch.backends.quantized.engine = backend

    config_path = logdir / "configs" / "_config.json"
    checkpoint_path = logdir / "checkpoints" / f"{checkpoint_name}.pth"
    logging.info("Load config")
    config: Dict[str, dict] = load_config(config_path)

    # Get expdir name
    config_expdir = Path(config["args"]["expdir"])
    # We will use copy of expdir from logs for reproducibility
    expdir = Path(logdir) / "code" / config_expdir.name

    logger.info("Import experiment and runner from logdir")
    experiment_fn, runner_fn = import_experiment_and_runner(expdir)
    experiment: ConfigExperiment = experiment_fn(config)

    logger.info(f"Load model state from checkpoints/{checkpoint_name}.pth")
    if stage is None:
        stage = list(experiment.stages)[0]

    model = experiment.get_model(stage)
    checkpoint = load_checkpoint(checkpoint_path)
    unpack_checkpoint(checkpoint, model=model)

    logger.info("Quantization is running...")
    quantized_model = quantization.quantize_dynamic(
        model.cpu(), qconfig_spec=qconfig_spec, dtype=dtype,
    )

    logger.info("Done")
    return quantized_model
