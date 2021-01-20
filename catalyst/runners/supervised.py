from typing import Any, Callable, List, Mapping, Tuple, Union
import logging

import torch

from catalyst.experiments.auto import AutoCallbackExperiment
from catalyst.runners.runner import Runner
from catalyst.typing import Device, RunnerModel

logger = logging.getLogger(__name__)


class SupervisedRunner(Runner):
    """Runner for experiments with supervised model."""

    def __init__(
        self,
        model: RunnerModel = None,
        device: Device = None,
        input_key: Any = "features",
        output_key: Any = "logits",
        input_target_key: str = "targets",
        experiment_fn: Callable = AutoCallbackExperiment,
    ):
        """
        Args:
            model: Torch model object
            device: Torch device
            input_key: Key in batch dict mapping for model input
            output_key: Key in output dict model output
                will be stored under
            input_target_key: Key in batch dict mapping for target
            experiment_fn: callable function,
                which defines default experiment type to use
                during ``.train`` and ``.infer`` methods.
        """
        super().__init__(model=model, device=device, experiment_fn=experiment_fn)

        self.input_key = input_key
        self.output_key = output_key
        self.target_key = input_target_key

        if isinstance(self.input_key, str):
            # when model expects value
            self._process_input = self._process_input_str
        elif isinstance(self.input_key, (list, tuple)):
            # when model expects tuple
            self._process_input = self._process_input_list
        elif self.input_key is None:
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
        elif self.output_key is None:
            # when model returns dict
            self._process_output = self._process_output_none
        else:
            raise NotImplementedError()

    def _process_input_str(self, batch: Mapping[str, Any], **kwargs):
        output = self.model(batch[self.input_key], **kwargs)
        return output

    def _process_input_list(self, batch: Mapping[str, Any], **kwargs):
        input = {key: batch[key] for key in self.input_key}  # noqa: WPS125
        output = self.model(**input, **kwargs)
        return output

    def _process_input_none(self, batch: Mapping[str, Any], **kwargs):
        output = self.model(**batch, **kwargs)
        return output

    def _process_output_str(self, output: torch.Tensor):
        output = {self.output_key: output}
        return output

    def _process_output_list(self, output: Union[Tuple, List]):
        output = {key: value for key, value in zip(self.output_key, output)}
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

    def _handle_device(self, batch: Mapping[str, Any]):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 2
            batch = {self.input_key: batch[0], self.target_key: batch[1]}
        batch = super()._handle_device(batch)
        return batch

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        """
        Inner method to handle specified data batch.
        Used to make a train/valid/infer stage during Experiment run.

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
        """
        self.output = self.forward(batch)

    @torch.no_grad()
    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Run model inference on specified data batch.

        .. warning::
            You should not override this method. If you need specific model
            call, override forward() method

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
            **kwargs: additional kwargs to pass to the model

        Returns:
            Mapping[str, Any]: model output dictionary
        """
        batch = self._handle_device(batch)
        output = self.forward(batch, **kwargs)
        return output


__all__ = ["SupervisedRunner"]
