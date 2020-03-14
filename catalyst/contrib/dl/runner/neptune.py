from pathlib import Path
import warnings

from deprecation import DeprecatedWarning

import neptune

from catalyst.dl import utils
from catalyst.dl.core import Experiment, Runner
from catalyst.dl.experiment import ConfigExperiment
from catalyst.dl.runner import SupervisedRunner

warnings.simplefilter("always")


class NeptuneRunner(Runner):
    """
    Runner wrapper with Neptune integration hooks.
    Read about Neptune here https://neptune.ai

    Examples:
        Initialize runner::

            from catalyst.dl import SupervisedNeptuneRunner
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
                       "project_qualified_name": "shared/catalyst-integration",
                       "api_token": "ANONYMOUS",  # api key,
                    },
                    "create_experiment": {
                        "name": "catalyst-example", # experiment name
                        "params": {"epoch_nr":10}, # immutable
                        "properties": {"data_source": "cifar10"} , # mutable
                        "tags": ["resnet", "no-augmentations"],
                        "upload_source_files": ["**/*.py"] # grep-like
                    }
                })

        You can see an example experiment here:
        https://ui.neptune.ai/o/shared/org/catalyst-integration/e/CAT-3/logs

        You can log your experiments there without registering.
        Just use "ANONYMOUS" token::

            runner.train(
                ...
                monitoring_params={
                    "init": {
                       "project_qualified_name": "shared/catalyst-integration",
                        "api_token": "ANONYMOUS",  # api key,
                    },
                    ...
                })

    """
    def _init(
        self,
        log_on_batch_end: bool = False,
        log_on_epoch_end: bool = True,
    ):
        super()._init()
        the_warning = DeprecatedWarning(
            self.__class__.__name__,
            deprecated_in="20.03",
            removed_in="20.04",
            details="Use NeptuneLogger instead."
        )
        warnings.warn(the_warning, category=DeprecationWarning, stacklevel=2)
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

    def _pre_experiment_hook(self, experiment: Experiment):
        monitoring_params = experiment.monitoring_params
        monitoring_params["dir"] = str(Path(experiment.logdir).absolute())

        neptune.init(**monitoring_params["init"])

        self._neptune_experiment = neptune.create_experiment(
            **monitoring_params["create_experiment"]
        )

        log_on_batch_end: bool = \
            monitoring_params.pop("log_on_batch_end", False)
        log_on_epoch_end: bool = \
            monitoring_params.pop("log_on_epoch_end", True)

        self._init(
            log_on_batch_end=log_on_batch_end,
            log_on_epoch_end=log_on_epoch_end,
        )

        self._neptune_experiment.set_property(
            "log_on_batch_end", self.log_on_batch_end
        )
        self._neptune_experiment.set_property(
            "log_on_epoch_end", self.log_on_epoch_end
        )

        if isinstance(experiment, ConfigExperiment):
            exp_config = utils.flatten_dict(experiment.stages_config)
            for name, value in exp_config.items():
                self._neptune_experiment.set_property(name, value)

    def _post_experiment_hook(self, experiment: Experiment):
        # @TODO: add params for artefacts logging
        # logdir_src = Path(experiment.logdir)
        # self._neptune_experiment.set_property("logdir", logdir_src)
        #
        # checkpoints_src = logdir_src.joinpath("checkpoints")
        # self._neptune_experiment.log_artifact(checkpoints_src)
        self._neptune_experiment.stop()

    def _run_batch(self, batch):
        super()._run_batch(batch=batch)
        if self.log_on_batch_end:
            mode = self.state.loader_name
            metrics = self.state.batch_metrics

            for name, value in metrics.items():
                self._neptune_experiment.log_metric(
                    f"batch_{mode}_{name}", value
                )

    def _run_epoch(self, stage: str, epoch: int):
        super()._run_epoch(stage=stage, epoch=epoch)
        if self.log_on_epoch_end:
            mode_metrics = utils.split_dict_to_subdicts(
                dct=self.state.epoch_metrics,
                prefixes=list(self.state.loaders.keys()),
                extra_key="_base",
            )
            for mode, metrics in mode_metrics.items():
                for name, value in metrics.items():
                    self._neptune_experiment.log_metric(
                        f"epoch_{mode}_{name}", value
                    )

    def run_experiment(self, experiment: Experiment):
        self._pre_experiment_hook(experiment=experiment)
        super().run_experiment(experiment=experiment)
        self._post_experiment_hook(experiment=experiment)


class SupervisedNeptuneRunner(NeptuneRunner, SupervisedRunner):
    pass


__all__ = ["NeptuneRunner", "SupervisedNeptuneRunner"]
