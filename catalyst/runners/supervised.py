from typing import Any, List, Mapping, Tuple, Union
from collections import OrderedDict

import torch

from catalyst.callbacks.backward import BackwardCallback
from catalyst.callbacks.criterion import CriterionCallback
from catalyst.callbacks.optimizer import OptimizerCallback
from catalyst.callbacks.scheduler import SchedulerCallback
from catalyst.core.callback import (
    Callback,
    IBackwardCallback,
    ICriterionCallback,
    IOptimizerCallback,
    ISchedulerCallback,
)
from catalyst.core.engine import Engine
from catalyst.core.misc import callback_isinstance, sort_callbacks_by_order
from catalyst.core.runner import IRunner
from catalyst.runners.runner import Runner
from catalyst.typing import RunnerModel, TorchCriterion, TorchOptimizer, TorchScheduler


class ISupervisedRunner(Runner):
    """IRunner for experiments with supervised model.

    Args:
        input_key: key in ``runner.batch`` dict mapping for model input
        output_key: key for ``runner.batch`` to store model output
        target_key: key in ``runner.batch`` dict mapping for target
        loss_key: key for ``runner.batch_metrics`` to store criterion loss output

    Abstraction, please check out implementations for more details:

        - :py:mod:`catalyst.runners.runner.SupervisedRunner`

    .. note::
        ISupervisedRunner contains only the logic with batch handling.


    ISupervisedRunner logic pseudocode:

    .. code-block:: python

        batch = {"input_key": tensor, "target_key": tensor}
        output = model(batch["input_key"])
        batch["output_key"] = output
        loss = criterion(batch["output_key"], batch["target_key"])
        batch_metrics["loss_key"] = loss



    .. note::
        Please follow the `minimal examples`_ sections for use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505

    """

    def __init__(
        self,
        input_key: Any = "features",
        output_key: Any = "logits",
        target_key: str = "targets",
        loss_key: str = "loss",
    ):
        """Init."""
        IRunner.__init__(self)

        self._input_key = input_key
        self._output_key = output_key
        self._target_key = target_key
        self._loss_key = loss_key

        if isinstance(self._input_key, str):
            # when model expects value
            self._process_input = self._process_input_str
        elif isinstance(self._input_key, (list, tuple)):
            # when model expects tuple
            self._process_input = self._process_input_list
        elif self._input_key is None:
            # when model expects dict
            self._process_input = self._process_input_none
        else:
            raise NotImplementedError()

        if isinstance(output_key, str):
            # when model returns value
            self._process_output = self._process_output_str
        elif isinstance(output_key, (list, tuple)):
            # when model returns tuple
            self._process_output = self._process_output_list
        elif self._output_key is None:
            # when model returns dict
            self._process_output = self._process_output_none
        else:
            raise NotImplementedError()

    def _process_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 2
            batch = {self._input_key: batch[0], self._target_key: batch[1]}
        return batch

    def _process_input_str(self, batch: Mapping[str, Any], **kwargs):
        output = self.model(batch[self._input_key], **kwargs)
        return output

    def _process_input_list(self, batch: Mapping[str, Any], **kwargs):
        input = {key: batch[key] for key in self._input_key}
        output = self.model(**input, **kwargs)
        return output

    def _process_input_none(self, batch: Mapping[str, Any], **kwargs):
        output = self.model(**batch, **kwargs)
        return output

    def _process_output_str(self, output: torch.Tensor):
        output = {self._output_key: output}
        return output

    def _process_output_list(self, output: Union[Tuple, List]):
        output = {key: value for key, value in zip(self._output_key, output)}
        return output

    def _process_output_none(self, output: Mapping[str, Any]):
        return output

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
        output = self._process_input(batch, **kwargs)
        output = self._process_output(output)
        return output

    def on_batch_start(self, runner: "IRunner"):
        """Event handler."""
        self.batch = self._process_batch(self.batch)
        super().on_batch_start(runner)

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        """
        Inner method to handle specified data batch.
        Used to make a train/valid/infer step during Experiment run.

        Args:
            batch: dictionary with data batches from DataLoader.
        """
        self.batch = {**batch, **self.forward(batch)}


class SupervisedRunner(ISupervisedRunner, Runner):
    """Runner for experiments with supervised model.

    Args:
        model: Torch model instance
        engine: Engine instance
        input_key: key in ``runner.batch`` dict mapping for model input
        output_key: key for ``runner.batch`` to store model output
        target_key: key in ``runner.batch`` dict mapping for target
        loss_key: key for ``runner.batch_metrics`` to store criterion loss output

    SupervisedRunner logic pseudocode:

    .. code-block:: python

        batch = {"input_key": tensor, "target_key": tensor}
        output = model(batch["input_key"])
        batch["output_key"] = output
        loss = criterion(batch["output_key"], batch["target_key"])
        batch_metrics["loss_key"] = loss

    .. note::
        Please follow the `minimal examples`_ sections for use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
    """

    def __init__(
        self,
        model: RunnerModel = None,
        engine: Engine = None,
        input_key: Any = "features",
        output_key: Any = "logits",
        target_key: str = "targets",
        loss_key: str = "loss",
    ):
        """Init."""
        ISupervisedRunner.__init__(
            self,
            input_key=input_key,
            output_key=output_key,
            target_key=target_key,
            loss_key=loss_key,
        )
        Runner.__init__(self, model=model, engine=engine)

    @torch.no_grad()
    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Run model inference on specified data batch.

        .. warning::
            You should not override this method.
            If you need specific model call, override runner.forward() method.

        Args:
            batch: dictionary with data batch from DataLoader.
            **kwargs: additional kwargs to pass to the model

        Returns:
            Mapping[str, Any]: model output dictionary
        """
        batch = self._process_batch(batch)
        output = self.forward(batch, **kwargs)
        return output

    def get_callbacks(self) -> "OrderedDict[str, Callback]":
        """Returns the callbacks for the experiment."""
        callbacks = sort_callbacks_by_order(super().get_callbacks())
        callback_exists = lambda callback_fn: any(
            callback_isinstance(x, callback_fn) for x in callbacks.values()
        )
        if isinstance(self._criterion, TorchCriterion) and not callback_exists(
            ICriterionCallback
        ):
            callbacks["_criterion"] = CriterionCallback(
                input_key=self._output_key,
                target_key=self._target_key,
                metric_key=self._loss_key,
            )
        if isinstance(self._optimizer, TorchOptimizer) and not callback_exists(
            IBackwardCallback
        ):
            callbacks["_backward"] = BackwardCallback(metric_key=self._loss_key)
        if isinstance(self._optimizer, TorchOptimizer) and not callback_exists(
            IOptimizerCallback
        ):
            callbacks["_optimizer"] = OptimizerCallback(metric_key=self._loss_key)
        if isinstance(self._scheduler, TorchScheduler) and not callback_exists(
            ISchedulerCallback
        ):
            callbacks["_scheduler"] = SchedulerCallback(
                loader_key=self._valid_loader, metric_key=self._valid_metric
            )
        return callbacks


__all__ = ["ISupervisedRunner", "SupervisedRunner"]
