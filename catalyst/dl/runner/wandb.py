from typing import Dict, List
import os
import shutil
from pathlib import Path
import wandb

from catalyst.dl.core import Runner, Experiment
from catalyst.dl.experiment import ConfigExperiment
from catalyst.dl import utils
from .supervised import SupervisedRunner


class WandbRunner(Runner):
    """
    Runner wrapper with wandb integration hooks.
    """
    @staticmethod
    def _log_metrics(metrics: Dict, mode: str, suffix: str = ""):
        metrics = {
            f"{key}/{mode}{suffix}": value
            for key, value in metrics.items()
        }
        wandb.log(metrics)

    def _init(
        self,
        log_on_batch_end: bool = False,
        log_on_epoch_end: bool = True,
        checkpoints_glob: List = None,
    ):
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end
        self.checkpoints_glob = checkpoints_glob or ["best.pth", "last.pth"]

        if (self.log_on_batch_end and not self.log_on_epoch_end) \
                or (not self.log_on_batch_end and self.log_on_epoch_end):
            self.batch_log_suffix = ""
            self.epoch_log_suffix = ""
        else:
            self.batch_log_suffix = "_batch"
            self.epoch_log_suffix = "_epoch"

    def _pre_experiment_hook(self, experiment: Experiment):
        monitoring_params = experiment.monitoring_params
        monitoring_params["dir"] = str(Path(experiment.logdir).absolute())

        log_on_batch_end: bool = \
            monitoring_params.pop("log_on_batch_end", False)
        log_on_epoch_end: bool = \
            monitoring_params.pop("log_on_epoch_end", True)
        checkpoints_glob: List[str] = \
            monitoring_params.pop("checkpoints_glob", None)
        self._init(
            log_on_batch_end=log_on_batch_end,
            log_on_epoch_end=log_on_epoch_end,
            checkpoints_glob=checkpoints_glob,
        )
        if isinstance(experiment, ConfigExperiment):
            exp_config = utils.flatten_dict(experiment.stages_config)
            wandb.init(**monitoring_params, config=exp_config)
        else:
            wandb.init(**monitoring_params)

    def _post_experiment_hook(self, experiment: Experiment):
        logdir_src = Path(experiment.logdir)
        logdir_dst = wandb.run.dir

        exclude = ["wandb", "checkpoints"]
        logdir_files = list(logdir_src.glob("*"))
        logdir_files = list(filter(
            lambda x: all(z not in str(x) for z in exclude),
            logdir_files))

        for subdir in logdir_files:
            if subdir.is_dir():
                os.makedirs(f"{logdir_dst}/{subdir.name}", exist_ok=True)
                shutil.rmtree(f"{logdir_dst}/{subdir.name}")
                shutil.copytree(
                    f"{str(subdir.absolute())}",
                    f"{logdir_dst}/{subdir.name}")
            else:
                shutil.copy2(
                    f"{str(subdir.absolute())}",
                    f"{logdir_dst}/{subdir.name}")

        checkpoints_src = logdir_src.joinpath("checkpoints")
        checkpoints_dst = Path(wandb.run.dir).joinpath("checkpoints")
        os.makedirs(checkpoints_dst, exist_ok=True)

        checkpoint_paths = []
        for glob in self.checkpoints_glob:
            checkpoint_paths.extend(list(checkpoints_src.glob(glob)))
        checkpoint_paths = list(set(checkpoint_paths))
        for checkpoint_path in checkpoint_paths:
            shutil.copy2(
                f"{str(checkpoint_path.absolute())}",
                f"{checkpoints_dst}/{checkpoint_path.name}")

    def _run_batch(self, batch):
        super()._run_batch(batch=batch)
        if self.log_on_batch_end:
            mode = self.state.loader_name
            metrics = self.state.metrics.batch_values
            self._log_metrics(
                metrics=metrics,
                mode=mode,
                suffix=self.batch_log_suffix
            )

    def _run_epoch(self, loaders):
        super()._run_epoch(loaders=loaders)
        if self.log_on_epoch_end:
            for mode, metrics in self.state.metrics.epoch_values.items():
                self._log_metrics(
                    metrics=metrics,
                    mode=mode,
                    suffix=self.epoch_log_suffix
                )

    def run_experiment(
        self,
        experiment: Experiment,
        check: bool = False
    ):
        self._pre_experiment_hook(experiment=experiment)
        super().run_experiment(experiment=experiment, check=check)
        self._post_experiment_hook(experiment=experiment)


class SupervisedWandbRunner(WandbRunner, SupervisedRunner):
    pass


__all__ = ["WandbRunner", "SupervisedWandbRunner"]
