from typing import List  # isort:skip
from pathlib import Path

import neptune

from catalyst.dl import utils
from catalyst.dl.core import Experiment, Runner
from catalyst.dl.experiment import ConfigExperiment
from catalyst.dl.runner import SupervisedRunner


class NeptuneRunner(Runner):
    """
    Runner wrapper with Neptune integration hooks.
    Read about Neptune here https://neptune.ml

    Examples:
        Initialize runner::

            from catalyst.contrib.runner.neptune import SupervisedNeptuneRunner
            runner = SupervisedNeptuneRunner()

        Pass `monitoring_params` and train model::

            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                loaders=loaders,
                logdir=logdir,
                num_epochs=num_epochs,
                verbose=True,
                monitoring_params={
                    "init": {
                    "project_qualified_name": "hapyp-user/examples",
                    "api_token": None, # api key, keep in NEPTUNE_API_TOKEN
                },
                    "create_experiment": {
                        "name": "catalyst-example", # experiment name
                        "params": {"epoch_nr":10}, # immutable
                        "properties": {"data_source": "cifar10"} , # mutable
                        "tags": ["resnet", "no-augmentations"],
                        "upload_source_files": ["**/*.py"] # grep-like
                    }
                })
    """

    def _init(
        self,
        log_on_batch_end: bool = True,
        log_on_epoch_end: bool = True,
        checkpoints_glob: List = None
    ):

        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end
        self.checkpoints_glob = checkpoints_glob

    def _pre_experiment_hook(self, experiment: Experiment):
        monitoring_params = experiment.monitoring_params
        monitoring_params["dir"] = str(Path(experiment.logdir).absolute())

        neptune.init(**monitoring_params["init"])

        self._neptune_experiment = neptune.create_experiment(
            **monitoring_params["create_experiment"])

        log_on_batch_end: bool = \
            monitoring_params.pop("log_on_batch_end", True)
        log_on_epoch_end: bool = \
            monitoring_params.pop("log_on_epoch_end", True)
        checkpoints_glob: List[str] = \
            monitoring_params.pop("checkpoints_glob", None)

        self._init(
            log_on_batch_end=log_on_batch_end,
            log_on_epoch_end=log_on_epoch_end,
            checkpoints_glob=checkpoints_glob,
        )

        self._neptune_experiment.set_property(
            "log_on_batch_end",
            self.log_on_batch_end
        )
        self._neptune_experiment.set_property(
            "log_on_epoch_end",
            self.log_on_epoch_end
        )
        self._neptune_experiment.set_property(
            "checkpoints_glob",
            self.checkpoints_glob
        )

        if isinstance(experiment, ConfigExperiment):
            exp_config = utils.flatten_dict(experiment.stages_config)
            for name, value in exp_config.items():
                self._neptune_experiment.set_property(name, value)

    def _post_experiment_hook(self, experiment: Experiment):
        logdir_src = Path(experiment.logdir)
        self._neptune_experiment.set_property("logdir", logdir_src)

        checkpoints_src = logdir_src.joinpath("checkpoints")
        self._neptune_experiment.log_artifact(checkpoints_src)
        self._neptune_experiment.stop()

    def _run_batch(self, batch):
        super()._run_batch(batch=batch)
        if self.log_on_batch_end:
            mode = self.state.loader_name
            metrics = self.state.metric_manager.batch_values

            for name, value in metrics.items():
                self._neptune_experiment.log_metric(
                    f"batch_{mode}_{name}",
                    value
                )

    def _run_epoch(self, stage: str, epoch: int):
        super()._run_epoch(stage=stage, epoch=epoch)
        if self.log_on_epoch_end:
            mode = self.state.loader_name
            metrics = self.state.metric_manager.batch_values

            for name, value in metrics.items():
                self._neptune_experiment.log_metric(
                    f"epoch_{mode}_{name}",
                    value
                )

    def run_experiment(
        self,
        experiment: Experiment,
        check: bool = False
    ):
        self._pre_experiment_hook(experiment=experiment)
        super().run_experiment(experiment=experiment, check=check)
        self._post_experiment_hook(experiment=experiment)


class SupervisedNeptuneRunner(NeptuneRunner, SupervisedRunner):
    pass


__all__ = ["NeptuneRunner", "SupervisedNeptuneRunner"]
