from typing import Dict, Optional, List, Union, Any, TYPE_CHECKING, NoReturn
import os
import pickle
import warnings

import numpy as np

from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if SETTINGS.aim_required:
    import aim
if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class AimLogger(ILogger):
    """Aim experimentation tracking for logging hypeparametrs, metrics .

    Aim documentation: https://aimstack.readthedocs.io/en/latest/.
    
    Args:
        experiment: Name of the experiment in Aim to log to.
        run_hash: Run hash.
        exclude: Name of key to exclude from logging.
        log_batch_metrics: boolean flag to log batch metrics
            (default: SETTINGS.log_batch_metrics or False).
        log_epoch_metrics: boolean flag to log epoch metrics
            (default: SETTINGS.log_epoch_metrics or True).

    Python API examples:
        .. code-block:: python
            from catalyst import dl
            runner = dl.SupervisedRunner()
            runner.train(
                ...,
                loggers={"aim": dl.AimLogger(experiment_name="test_exp")}
            )
        .. code-block:: python
            from catalyst import dl
            class CustomRunner(dl.IRunner):
                # ...
                def get_loggers(self):
                    return {
                        "console": dl.ConsoleLogger(),
                        "aim": dl.AimLogger(experiment_name="test_exp")
                    }
                # ...
            runner = CustomRunner().run()
    """

    def __init__(
        self,
        experiment_name: str,
        run_hash: Optional[str] = None,
        exclude: Optional[str] = None,
        repo: Optional[Union[str, aim.Repo]] = '.',
        log_batch_metrics: bool = SETTINGS.log_batch_metrics,
        log_epoch_metrics: bool = SETTINGS.log_epoch_metrics,
        **kwargs,
    ):
        self.experiment_name = experiment_name
        self.run_hash = run_hash
        self.exclude = exclude
        self.kwargs = kwargs

        super().__init__(
            log_batch_metrics=log_batch_metrics,
            log_epoch_metrics=log_epoch_metrics,
        )

        self.exclude = [] if exclude is None else exclude
        self.repo = repo

        self.aim_run = aim.Run(experiment = experiment_name, run_hash = run_hash, repo = repo)


    @property
    def logger(self):
        """Logger property."""
        return self.aim_run

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str,
        runner: "IRunner",
    ) -> NoReturn:
        """
        Log metrics to Aim.

        Args:
            metrics: Dict of metrics to log.
            scope: Scope of the metrics.
            runner: Runner instance.

        """

        if scope == "batch" and self.log_batch_metrics:
            metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(
                metrics=metrics,
                runner=runner,
                metric_type=runner.loader_key,
                scope=scope,
            )
        elif scope == "epoch" and self.log_epoch_metrics:
            for metric_type, metric_values in metrics.items():
                self._log_metrics(
                    metrics=metric_values,
                    runner=runner,
                    metric_type=metric_type,
                    scope=scope,
                )


    def _log_metrics(
        self,
        metrics: Dict[str, float],
        runner: "IRunner",
        metric_type: str,
        scope: str = "",
    ):
        # print("Metric Type:", metric_type)
        # print("Scope:", scope)
        # print("Metrics:", metrics)
        context, kwargs  = self._aim_context(runner, scope, metric_type)
        for metric_name, value in metrics.items():
            self.aim_run.track(value,name=metric_name, context=context,epoch=kwargs["epoch"])

    def log_artifact(
        self,
        tag: str,
        runner: "IRunner",
        artifact: object = None,
        path_to_artifact: str = None,
        scope: str = None,
        kind: str = "text",
        artifact_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Logs a local file or directory as an artifact to the logger."""
        
        if path_to_artifact:
            mode = "r" if kind == "text" else "rb"
            with open(path_to_artifact, mode) as f:
                artifact = f.read()


        kind_dict = {
            "audio": aim.Audio,
            "figure": aim.Figure,
            "image": aim.Image,
            "text": aim.Text,
        }
        value = kind_dict[kind](artifact, **artifact_kwargs)
        context, kwargs = self._aim_context(runner, scope)
        self.aim_run.track(value, tag, context=context, epoch=kwargs["epoch"])


    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        runner: "IRunner",
        scope: str = None,
        image_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Logs image to Aim for current scope on current step."""
        value = aim.Image(image, **image_kwargs)
        context, kwargs = self._aim_context(runner, scope)
        self.aim_run.track(value, tag, context=context, epoch=kwargs["epoch"])


    def log_hparams(self, hparams: Dict, runner: "IRunner" = None) -> None:
        """
        Logs parameters for current scope.
        Args:
            hparams: Parameters to log.
            runner: experiment runner
        """
        hparams = self._build_params_dict(hparams, self.exclude)

        # print(hparams)

        for k, v in hparams.items():
            self.aim_run[k] = v



    def _aim_context(self,
        runner: "IRunner",
        scope: Optional[str],
        metric_type: Optional[str] = None,
        all_scope_steps: bool = False,
    ):
        if metric_type is None:
            metric_type = runner.loader_key
        context = {}
        if metric_type is not None:
            context["metric_type"] = metric_type
        if scope is not None:
            context["metric_context"] = scope
        
        kwargs = {}
        if all_scope_steps or scope == "batch":
            kwargs["step"] = runner.batch_step
        if all_scope_steps or scope == "epoch" or scope == "loader":
            kwargs["epoch"] = runner.epoch_step

        return context, kwargs

    def close_log(self) -> NoReturn:
        """Close log."""
        self.aim_run.close()
        del self.aim_run

    def _build_params_dict(self,
        dictionary: Dict[str, Any],
        exclude: List[str],
    ):
        clear_dict = {}
        strap_dict = {}
        for name, value in dictionary.items():
            if name in exclude:
                continue

            if isinstance(value, dict):
                if name not in strap_dict:
                    strap_dict[name] = {}
                
                strap_dict[name] = self._build_params_dict(value, exclude)
            else:
                strap_dict[name] = value

        
        clear_dict.update(strap_dict)
        
        return clear_dict


__all__ = ["AimLogger"]
