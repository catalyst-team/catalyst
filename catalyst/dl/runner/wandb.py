from typing import Dict, List
import os
import shutil
from pathlib import Path
import logging

from catalyst.dl.core import Runner
from catalyst.dl.experiment import ConfigExperiment
from catalyst.dl import utils

logger = logging.getLogger(__name__)
try:
    import wandb
    WANDB_ENABLED = True
except ImportError:
    logger.warning(
        "wandb not available, switching to pickle. "
        "To install wandb, run `pip install wandb`."
    )
    WANDB_ENABLED = False


if WANDB_ENABLED:
    class WandbRunner(Runner):

        @staticmethod
        def _log_metrics(metrics: Dict, mode: str):
            metrics = {
                f"{mode}/{key}": value
                for key, value in metrics.items()
            }
            wandb.log(metrics)

        def _run_epoch(self, loaders):
            super()._run_epoch(loaders=loaders)
            for mode, metrics in self.state.metrics.epoch_values.items():
                self._log_metrics(metrics=metrics, mode=mode)

        def run_experiment(
            self,
            experiment: ConfigExperiment,
            check: bool = False
        ):
            monitoring_params = experiment.monitoring_params
            monitoring_params["dir"] = str(Path(experiment.logdir).absolute())
            checkpoints_glob: List[str] = \
                monitoring_params.pop(
                    "checkpoints_glob", ["best.pth", "last.pth"])

            flatten_config = utils.flatten_dict(experiment.stages_config)
            wandb.init(**monitoring_params, config=flatten_config)

            super().run_experiment(experiment=experiment, check=check)

            logdir_src = Path(experiment.logdir)
            logidr_dst = wandb.run.dir

            exclude = ["wandb", "checkpoints"]
            logdir_files = list(logdir_src.glob("*"))
            logdir_files = list(filter(
                lambda x: all(z not in str(x) for z in exclude),
                logdir_files))

            for subdir in logdir_files:
                if subdir.is_dir():
                    os.makedirs(f"{logidr_dst}/{subdir.name}", exist_ok=True)
                    shutil.rmtree(f"{logidr_dst}/{subdir.name}")
                    shutil.copytree(
                        f"{str(subdir.absolute())}",
                        f"{logidr_dst}/{subdir.name}")
                else:
                    shutil.copy2(
                        f"{str(subdir.absolute())}",
                        f"{logidr_dst}/{subdir.name}")

            checkpoints_src = logdir_src.joinpath("checkpoints")
            checkpoints_dst = Path(wandb.run.dir).joinpath("checkpoints")
            os.makedirs(checkpoints_dst, exist_ok=True)

            checkpoint_paths = []
            for glob in checkpoints_glob:
                checkpoint_paths.extend(list(checkpoints_src.glob(glob)))
            checkpoint_paths = list(set(checkpoint_paths))
            for checkpoint_path in checkpoint_paths:
                shutil.copy2(
                    f"{str(checkpoint_path.absolute())}",
                    f"{checkpoints_dst}/{checkpoint_path.name}")
