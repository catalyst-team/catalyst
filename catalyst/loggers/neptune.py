from typing import Dict

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if SETTINGS.neptune_required:
    import neptune.new as neptune


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
    """Neptune logger for parameters, metrics, images and other artifacts.

    Neptune documentation: https://docs.neptune.ai

    You can acquire api_token by following:
    https://docs.neptune.ai/getting-started/installation#authentication
    or, you can use special token 'ANONYMOUS' for testing without registration
    if not provided, Neptune will try to use environmental variable NEPTUNE_API_TOKEN

    If using 'ANONYMOUS' token, you can set project to 'common/catalyst-integration'
    for test purposes.

    Additional keyword arguments will be passed directly to neptune.init() function,
    see https://docs.neptune.ai/api-reference/neptune#init

    After creation of the logger without passing run parameter, a link to created run
    will be printed. You can also retrieve the run object by calling NeptuneLogger.run
    to access it directly

    Args:
        base_namespace: Optional. namespace within Neptune's Run to put all metric in
        api_token: Optional. Your Neptune API token. Use 'ANONYMOUS' for common, public access
        project: Optional. Name of your workspace in a form of 'workspaceName/projectName'
        run: Optional. Pass if you want to resume a Neptune run.

    Python API examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            ...,
            loggers={"neptune": dl.NeptuneLogger(
                base_namespace="catalyst-tests",
                api_token="ANONYMOUS",
                project="common/catalyst-integration")
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
                        base_namespace="catalyst-tests",
                        api_token="ANONYMOUS",
                        project="common/catalyst-integration")
                }
            # ...

        runner = CustomRunner().run()
    """

    def __init__(
        self, base_namespace=None, api_token=None, project=None, run=None, **neptune_run_kwargs
    ):
        if base_namespace is None:
            self.base_namespace = "experiment"
        else:
            self.base_namespace = base_namespace
        self._api_token = api_token
        self._project = project
        self._neptune_run_kwargs = neptune_run_kwargs
        if run is None:
            self.run = neptune.init(
                project=self._project, api_token=self._api_token, **self._neptune_run_kwargs
            )
        else:
            self.run = run
        try:
            import catalyst.__version__ as version

            self.run["source_code/integrations/neptune-catalyst"] = version
        except (ImportError, NameError, AttributeError):
            pass

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

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs batch, epoch and loader metrics to Neptune."""
        if scope == "batch":
            neptune_path = "/".join([self.base_namespace, stage_key, loader_key, scope])
            self._log_metrics(metrics=metrics, neptune_path=neptune_path, step=global_sample_step)
        elif scope == "loader":
            neptune_path = "/".join([self.base_namespace, stage_key, loader_key, scope])
            self._log_metrics(
                metrics=_prepare_metrics(metrics),
                neptune_path=neptune_path,
                step=global_epoch_step,
            )
        elif scope == "epoch":
            loader_key = "_epoch_"
            prepared_metrics = _prepare_metrics(metrics[loader_key])
            neptune_path = "/".join([self.base_namespace, stage_key, scope])
            if prepared_metrics:
                self._log_metrics(
                    metrics=prepared_metrics, neptune_path=neptune_path, step=global_epoch_step
                )
        elif scope == "stage":
            neptune_path = "/".join([self.base_namespace, stage_key])
            self._log_metrics(metrics=metrics, neptune_path=neptune_path, step=0)
        elif scope == "experiment" or scope is None:
            self._log_metrics(metrics=metrics, neptune_path=self.base_namespace, step=0)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs image to Neptune for current scope on current step."""
        if scope == "batch" or scope == "loader":
            neptune_path = "/".join(
                [self.base_namespace, stage_key, loader_key, scope, "_images", tag]
            )
            self._log_image(image, neptune_path)
        elif scope == "epoch":
            neptune_path = "/".join([self.base_namespace, stage_key, scope, "_images", tag])
            self._log_image(image, neptune_path)
        elif scope == "stage":
            neptune_path = "/".join([self.base_namespace, stage_key, "_images", tag])
            self._log_image(image, neptune_path)
        elif scope == "experiment" or scope is None:
            neptune_path = "/".join([self.base_namespace, "_images", tag])
            self._log_image(image, neptune_path)

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        """Logs hyper-parameters to Neptune."""
        if scope == "stage":
            self.run[f"{self.base_namespace}/{stage_key}/hparams"] = hparams
        elif scope == "experiment" or scope is None:
            self.run[f"{self.base_namespace}/hparams"] = hparams

    def log_artifact(
        self,
        tag: str,
        artifact: object = None,
        path_to_artifact: str = None,
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs arbitrary file (audio, video, csv, etc.) to Neptune."""
        if artifact is not None and path_to_artifact is not None:
            ValueError("artifact and path_to_artifact are mutually exclusive")
        if scope == "batch":
            neptune_path = "/".join(
                [
                    self.base_namespace,
                    stage_key,
                    loader_key,
                    scope,
                    "_artifacts",
                    "batch-" + str(global_batch_step),
                    tag,
                ]
            )
            self._log_artifact(artifact, path_to_artifact, neptune_path)
        elif scope == "loader":
            neptune_path = "/".join(
                [
                    self.base_namespace,
                    stage_key,
                    loader_key,
                    scope,
                    "_artifacts",
                    "epoch-" + str(global_epoch_step),
                    tag,
                ]
            )
            self._log_artifact(artifact, path_to_artifact, neptune_path)
        elif scope == "epoch":
            neptune_path = "/".join(
                [
                    self.base_namespace,
                    stage_key,
                    scope,
                    "_artifacts",
                    "epoch-" + str(global_epoch_step),
                    tag,
                ]
            )
            self._log_artifact(artifact, path_to_artifact, neptune_path)
        elif scope == "stage":
            neptune_path = "/".join(
                [
                    self.base_namespace,
                    stage_key,
                    "_artifacts",
                    "epoch-" + str(global_epoch_step),
                    tag,
                ]
            )
            self._log_artifact(artifact, path_to_artifact, neptune_path)
        elif scope == "experiment" or scope is None:
            neptune_path = "/".join(
                [self.base_namespace, "_artifacts", "epoch-" + str(global_epoch_step), tag]
            )
            self._log_artifact(artifact, path_to_artifact, neptune_path)

    def flush_log(self) -> None:
        """Flushes the loggers."""
        pass

    def close_log(self) -> None:
        """Closes the loggers."""
        self.run.wait()


__all__ = ["NeptuneLogger"]
