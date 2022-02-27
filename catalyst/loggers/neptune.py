from typing import Dict, TYPE_CHECKING

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if SETTINGS.neptune_required:
    import neptune.new as neptune
if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


def _prepare_metrics(metrics):
    conflict_keys = []
    processed_metrics = dict(metrics)
    for k in list(processed_metrics.keys()):
        if k.endswith("/std"):
            k_stripped = k[:-4]
            k_val = k_stripped + "/val"
            if k_val not in processed_metrics.keys():
                processed_metrics[k_val] = processed_metrics.pop(k_stripped)
    for k in processed_metrics.keys():
        for j in processed_metrics.keys():
            if j.startswith(k) and j != k and k not in conflict_keys:
                conflict_keys.append(k)
    for i in conflict_keys:
        processed_metrics[i + "_val"] = processed_metrics.pop(i)
    return processed_metrics


class NeptuneLogger(ILogger):
    """Neptune logger for parameters, metrics, images and other artifacts (videos, audio,
    model checkpoints, etc.).

    Neptune documentation:
    https://docs.neptune.ai/integrations-and-supported-tools/model-training/catalyst

    When the logger is created, link to the run in Neptune will be printed to stdout.
    It looks like this:
    https://ui.neptune.ai/common/catalyst-integration/e/CATALYST-1379

    To start with Neptune please check
    `Neptune getting-started docs <http://docs.neptune.ai/getting-started/installation>`_
    because you will need ``api_token`` and project to log your Catalyst runs to.

    .. note::
        You can use public api_token ``ANONYMOUS`` and set project to
        ``common/catalyst-integration`` for testing without registration.

    Args:
        base_namespace: Optional, ``str``, root namespace within Neptune's run.
          Default is "experiment".
        api_token: Optional, ``str``. Your Neptune API token. Read more about it in the
          `Neptune docs <http://docs.neptune.ai/getting-started/installation>`_.
        project: Optional, ``str``. Name of the project to log runs to.
          It looks like this: "my_workspace/my_project".
        run: Optional, pass Neptune run object if you want to continue logging
          to the existing run (resume run).
          Read more about it
          `here <https://docs.neptune.ai/how-to-guides/neptune-api/resume-run>`_.
        log_batch_metrics: boolean flag to log batch metrics
          (default: SETTINGS.log_batch_metrics or False).
        log_epoch_metrics: boolean flag to log epoch metrics
          (default: SETTINGS.log_epoch_metrics or True).
        neptune_run_kwargs: Optional, additional keyword arguments
          to be passed directly to the
          `neptune.init()  <https://docs.neptune.ai/api-reference/neptune#init>`_
          function.

    Python API examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            ...
            loggers={
                "neptune": dl.NeptuneLogger(
                    project="my_workspace/my_project",
                    tags=["pretraining", "retina"],
                )
            }
        )

    .. code-block:: python

        from catalyst import dl

        class CustomRunner(dl.IRunner):
            # ...

            def get_loggers(self):
                return {
                    "console": dl.ConsoleLogger(),
                    "neptune": dl.NeptuneLogger(
                        project="my_workspace/my_project"
                    )
                }
            # ...

        runner = CustomRunner().run()
    """

    def __init__(
        self,
        base_namespace=None,
        api_token=None,
        project=None,
        run=None,
        log_batch_metrics: bool = SETTINGS.log_batch_metrics,
        log_epoch_metrics: bool = SETTINGS.log_epoch_metrics,
        **neptune_run_kwargs,
    ):
        super().__init__(
            log_batch_metrics=log_batch_metrics, log_epoch_metrics=log_epoch_metrics
        )
        if base_namespace is None:
            self.base_namespace = "experiment"
        else:
            self.base_namespace = base_namespace
        self._api_token = api_token
        self._project = project
        self._neptune_run_kwargs = neptune_run_kwargs
        if run is None:
            self.run = neptune.init(
                project=self._project,
                api_token=self._api_token,
                **self._neptune_run_kwargs,
            )
        else:
            self.run = run
        try:
            import catalyst.__version__ as version

            self.run["source_code/integrations/neptune-catalyst"] = version
        except (ImportError, NameError, AttributeError):
            pass

    @property
    def logger(self):
        """Internal logger/experiment/etc. from the monitoring system."""
        return self.run

    def _log_metrics(self, metrics: Dict[str, float], neptune_path: str, step: int):
        for key, value in metrics.items():
            self.run[f"{neptune_path}/{key}"].log(value=float(value), step=step)

    def _log_image(self, image: np.ndarray, neptune_path: str):
        self.run[neptune_path].log(neptune.types.File.as_image(image))

    def _log_artifact(self, artifact: object, path_to_artifact: str, neptune_path: str):
        if artifact is not None:
            self.run[neptune_path].upload(neptune.types.File.as_pickle(artifact))
        elif path_to_artifact is not None:
            self.run[neptune_path].upload(path_to_artifact)

    def log_artifact(
        self,
        tag: str,
        runner: "IRunner",
        artifact: object = None,
        path_to_artifact: str = None,
        scope: str = None,
    ) -> None:
        """Logs arbitrary file (audio, video, csv, etc.) to Neptune."""
        if artifact is not None and path_to_artifact is not None:
            ValueError("artifact and path_to_artifact are mutually exclusive")
        if scope == "batch":
            neptune_path = "/".join(
                [
                    self.base_namespace,
                    "_artifacts",
                    f"epoch-{runner.epoch_step:04d}",
                    f"loader-{runner.loader_key}",
                    f"batch-{runner.batch_step:04d}",
                    tag,
                ]
            )
        elif scope == "loader":
            neptune_path = "/".join(
                [
                    self.base_namespace,
                    "_artifacts",
                    f"epoch-{runner.epoch_step:04d}",
                    f"loader-{runner.loader_key}",
                    tag,
                ]
            )
        elif scope == "epoch":
            neptune_path = "/".join(
                [
                    self.base_namespace,
                    "_artifacts",
                    f"epoch-{runner.epoch_step:04d}",
                    tag,
                ]
            )
        elif scope == "experiment" or scope is None:
            neptune_path = "/".join([self.base_namespace, "_artifacts", tag])
        self._log_artifact(artifact, path_to_artifact, neptune_path)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        runner: "IRunner",
        scope: str = None,
    ) -> None:
        """Logs image to Neptune for current scope on current step."""
        if scope == "batch" or scope == "loader":
            log_path = "/".join(
                [
                    self.base_namespace,
                    "_images",
                    f"epoch-{runner.epoch_step:04d}",
                    f"loader-{runner.loader_key}",
                    tag,
                ]
            )
        elif scope == "epoch":
            log_path = "/".join(
                [self.base_namespace, "_images", f"epoch-{runner.epoch_step:04d}", tag]
            )
        elif scope == "experiment" or scope is None:
            log_path = "/".join([self.base_namespace, "_images", tag])
        self._log_image(image, log_path)

    def log_hparams(self, hparams: Dict, runner: "IRunner" = None) -> None:
        """Logs hyper-parameters to Neptune."""
        self.run[f"{self.base_namespace}/hparams"] = hparams

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str,
        runner: "IRunner",
    ) -> None:
        """Logs batch, epoch and loader metrics to Neptune."""
        if scope == "batch" and self.log_batch_metrics:
            neptune_path = "/".join([self.base_namespace, runner.loader_key, scope])
            self._log_metrics(
                metrics=metrics, neptune_path=neptune_path, step=runner.sample_step
            )
        elif scope == "loader" and self.log_epoch_metrics:
            neptune_path = "/".join([self.base_namespace, runner.loader_key, scope])
            self._log_metrics(
                metrics=_prepare_metrics(metrics),
                neptune_path=neptune_path,
                step=runner.epoch_step,
            )
        elif scope == "epoch" and self.log_epoch_metrics:
            loader_key = "_epoch_"
            prepared_metrics = _prepare_metrics(metrics[loader_key])
            neptune_path = "/".join([self.base_namespace, scope])
            if prepared_metrics:
                self._log_metrics(
                    metrics=prepared_metrics,
                    neptune_path=neptune_path,
                    step=runner.epoch_step,
                )
        elif scope == "experiment" or scope is None:
            self._log_metrics(metrics=metrics, neptune_path=self.base_namespace, step=0)

    def flush_log(self) -> None:
        """Flushes the loggers."""
        pass

    def close_log(self, scope: str = None) -> None:
        """Closes the loggers."""
        self.run.wait()


__all__ = ["NeptuneLogger"]
