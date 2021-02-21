from pathlib import Path
from typing import Dict

from catalyst.utils import (
    load_config,
    prepare_config_api_components,
    load_checkpoint,
    unpack_checkpoint
)


def load_model(logdir: Path, checkpoint_name: str = "best", stage: str = None):
    checkpoint_path = logdir / "checkpoints" / f"{checkpoint_name}.pth"


    experiment, _ = load_experiment(logdir=logdir)

    if stage is None:
        stage = list(experiment.stages)[0]

    model = experiment.get_model(stage)
    checkpoint = load_checkpoint(checkpoint_path)
    unpack_checkpoint(checkpoint, model=model)
    return model


def load_experiment(logdir: Path):
    config_path = logdir / "configs" / "_config.json"
    config: Dict[str, dict] = load_config(config_path)

    # Get expdir name
    config_expdir = Path(config["args"]["expdir"])
    # We will use copy of expdir from logs for reproducibility
    expdir = Path(logdir) / "code" / config_expdir.name

    experiment, runner, _ = prepare_config_api_components(expdir=expdir, config=config)
    return experiment, runner


def get_model_file_name(
    prefix: str, 
    method_name: str = "forward",
    mode: str = "train",
    requires_grad: bool = False,
    opt_level: str = None,
    additional_string: str = None,
):
    file_name = prefix
    if additional_string is not None:
        file_name += f"-{additional_string}"
    if method_name != "forward":
        file_name += f"-{method_name}"
       
    if mode == "train":
        file_name += "-in_train"

    if requires_grad:
        file_name += "-with_grad"

    if opt_level is not None:
        file_name += "-opt_{opt_level}"

    file_name += ".pth"

    return file_name
