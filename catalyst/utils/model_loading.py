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


    experiment, runner = load_experiment(logdir=logdir)

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
