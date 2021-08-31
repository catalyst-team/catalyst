from typing import Any, Mapping
from collections import OrderedDict

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from catalyst.callbacks.criterion import CriterionCallback, ICriterionCallback
from catalyst.callbacks.optimizer import IOptimizerCallback, OptimizerCallback
from catalyst.callbacks.scheduler import ISchedulerCallback, SchedulerCallback
from catalyst.core.callback import Callback
from catalyst.core.misc import callback_isinstance
from catalyst.core.runner import IRunner
from catalyst.engines import IEngine
from catalyst.runners.runner import Runner
from catalyst.typing import Criterion, Optimizer, RunnerModel, Scheduler


class IContrastiveRunner(IRunner):
    """IRunner for experiments with contrastive model.

    Args:
        input_key: key in ``runner.batch`` dict mapping for model input
        output_key: key for ``runner.batch`` to store model output
        target_key: key in ``runner.batch`` dict mapping for target
        loss_key: key for ``runner.batch_metrics`` to store criterion loss output
        projection_key: key for ``runner.batch`` to store model projection
        embedding_key: key for `runner.batch`` to store model embeddings

    Abstraction, please check out implementations for more details:

        - :py:mod:`catalyst.runners.contrastive.ContrastiveRunner`

    .. note::
        IContrastiveRunner contains only the logic with batch handling.


    ISupervisedRunner logic pseudocode:

    .. code-block:: python

        batch = {...}

    .. note::
        Please follow the `minimal examples`_ sections for use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples

    """

    def __init__(
        self,
        target_key: str = "targets",
        loss_key: str = "loss",
        augemention_prefix: str = "aug",
        projection_prefix: str = "projections",
        embedding_prefix: str = "embeddings",
    ):
        """Init."""
        IRunner.__init__(self)

        self._target_key = target_key
        self._loss_key = loss_key
        self._projection_prefix = projection_prefix
        self._augemention_prefix = augemention_prefix
        self._embedding_prefix = embedding_prefix

    def _process_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 3
            batch = {
                f"{self._augemention_prefix}_1": batch[0],
                f"{self._augemention_prefix}_2": batch[1],
                self._target_key: batch[2],
            }
        return batch

    def _process_input(self, batch: Mapping[str, Any], **kwargs):
        embedding1, projection1 = self.model(batch[f"{self._augemention_prefix}_left"], **kwargs)
        embedding2, projection2 = self.model(batch[f"{self._augemention_prefix}_right"], **kwargs)

        batch = {
            **batch,
            f"{self._projection_prefix}_left": projection1,
            f"{self._projection_prefix}_right": projection2,
            f"{self._embedding_prefix}_left": embedding1,
            f"{self._embedding_prefix}_right": embedding2,
        }
        return batch

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Forward method for your Runner.
        Should not be called directly outside of runner.
        If your model has specific interface, override this method to use it

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoaders.
            **kwargs: additional parameters to pass to the model

        Returns:
            dict with model output batch
        """
        return self._process_input(batch, **kwargs)


class ContrastiveRunner(IContrastiveRunner, Runner):
    """Runner for experiments with contrastive model."""

    def __init__(
        self,
        model: RunnerModel = None,
        engine: IEngine = None,
        target_key: str = "targets",
        loss_key: str = "loss",
        augemention_key: str = "aug",
        projection_key: str = "projections",
        embedding_key: str = "embeddings",
    ):
        """Init."""
        IContrastiveRunner.__init__(
            self,
            target_key=target_key,
            loss_key=loss_key,
            augemention_key=augemention_key,
            projection_key=projection_key,
            embedding_key=embedding_key,
        )
        Runner.__init__(self, model=model, engine=engine)

    @torch.no_grad()
    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Run model inference on specified data batch.

        .. warning::
            You should not override this method. If you need specific model
            call, override forward() method

        Args:
            batch: dictionary with data batch from DataLoader.
            **kwargs: additional kwargs to pass to the model

        Returns:
            Mapping[str, Any]: model output dictionary
        """
        batch = self._process_batch(batch)
        batch = self.engine.sync_device(tensor_or_module=batch)
        output = self.forward(batch, **kwargs)
        return output

    def handle_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Run model forward on specified data batch.

        .. warning::
            You should not override this method. If you need specific model
            call, override forward() method

        Args:
            batch: dictionary with data batch from DataLoader.
            **kwargs: additional kwargs to pass to the model

        Returns:
            Mapping[str, Any]: model output dictionary
        """
        batch = self._process_batch(batch)
        batch = self.engine.sync_device(tensor_or_module=batch)
        output = self.forward(batch, **kwargs)
        return output

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """Prepares the callbacks for selected stage.

        Args:
            stage: stage name

        Returns:
            dictionary with stage callbacks
        """
        # I took it from supervised runner should be remade
        callbacks = super().get_callbacks(stage=stage)
        is_callback_exists = lambda callback_fn: any(
            callback_isinstance(x, callback_fn) for x in callbacks.values()
        )
        if isinstance(self._criterion, Criterion) and not is_callback_exists(ICriterionCallback):
            callbacks["_criterion"] = CriterionCallback(
                input_key=f"{self._projection_prefix}_1",
                target_key=f"{self._projection_prefix}_2",
                metric_key=self._loss_key,
            )
        if isinstance(self._optimizer, Optimizer) and not is_callback_exists(IOptimizerCallback):
            callbacks["_optimizer"] = OptimizerCallback(metric_key=self._loss_key)
        if isinstance(self._scheduler, (Scheduler, ReduceLROnPlateau)) and not is_callback_exists(
            ISchedulerCallback
        ):
            callbacks["_scheduler"] = SchedulerCallback(
                loader_key=self._valid_loader, metric_key=self._valid_metric
            )
        return callbacks
