from typing import Any, Callable, List, Mapping, Tuple, Union
import logging

import torch

from catalyst.experiments.auto import AutoCallbackExperiment
from catalyst.runners.runner import Runner
from catalyst.typing import Device, RunnerModel

logger = logging.getLogger(__name__)


class MultiSupervisedRunner(Runner):
    """Runner for experiments with supervised models."""

    def __init__(
        self,
        model: RunnerModel = None,
        device: Device = None,
        models_keys: Mapping[str, Any] = None,
        experiment_fn: Callable = AutoCallbackExperiment,
    ):
        """

        Args:
            model: (RunnerModel) Torch model object
            device: (Device) Torch device
            models_keys: (Mapping[str, Any]) Key in batch dict mapping
                for model input, output, target
            experiment_fn: callable function,
                which defines default experiment type to use
                during ``.train`` and ``.infer`` methods.

        """
        super().__init__(
            model=model, device=device, experiment_fn=experiment_fn,
        )

        self.input_key = {}
        self.output_key = {}
        self.target_key = {}

        self._process_input = {}
        self._process_output = {}

        for model_name, model_keys in models_keys.items():
            self.input_key[model_name] = (
                model_keys["input_key"] if model_keys["input_key"] is not None else "features"
            )
            self.output_key[model_name] = (
                model_keys["output_key"] if model_keys["output_key"] is not None else "logits"
            )
            self.target_key[model_name] = (
                model_keys["target_key"] if model_keys["target_key"] is not None else "targets"
            )
            if isinstance(self.input_key[model_name], str):
                # when model expects value
                self._process_input[model_name] = self._process_input_str
            elif isinstance(self.input_key[model_name], (list, tuple)):
                # when model expects tuple
                self._process_input[model_name] = self._process_input_list
            elif self.input_key[model_name] is None:
                # when model expects dict
                self._process_input[model_name] = self._process_input_none
            else:
                raise NotImplementedError()
            if isinstance(self.output_key[model_name], str):
                # when model returns value
                self._process_output[model_name] = self._process_output_str
            elif isinstance(self.output_key[model_name], (list, tuple)):
                # when model returns tuple
                self._process_output[model_name] = self._process_output_list
            elif self.output_key[model_name] is None:
                # when model returns dict
                self._process_output[model_name] = self._process_output_none
            else:
                raise NotImplementedError()

    def _prepare_inner_state(self, *args, **kwargs):
        logdir = None if self.experiment is None else self.experiment.logdir
        super()._prepare_inner_state(*args, logdir=logdir, **kwargs)

    def _handle_device(self, batch: Mapping[str, Any]):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 2
            batch_dict = {}
            for _, input_key in self.input_key.items():
                batch_dict.setdefault(input_key, batch[0])
            for _, target_key in self.target_key.items():
                batch_dict.setdefault(target_key, batch[1])
            batch = batch_dict
        batch = super()._handle_device(batch)
        return batch

    def _process_input_str(self, model_name: str, batch: Mapping[str, Any], **kwargs):
        output = self.model[model_name](batch[self.input_key[model_name]], **kwargs)
        return output

    def _process_input_list(self, model_name: str, batch: Mapping[str, Any], **kwargs):
        input = {key: batch[key] for key in self.input_key[model_name]}
        output = self.model[model_name](**input, **kwargs)
        return output

    def _process_input_none(self, model_name: str, batch: Mapping[str, Any], **kwargs):
        output = self.model[model_name](**batch, **kwargs)
        return output

    def _process_output_str(self, model_name: str, output: torch.Tensor):
        output = {self.output_key[model_name]: output}
        return output

    def _process_output_list(self, model_name: str, output: Union[Tuple, List]):
        output = {key: value for key, value in zip(self.output_key[model_name], output)}
        return output

    def _process_output_none(self, model_name: str, output: Mapping[str, Any]):
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
        output = {}
        for model_name in self.model:
            output = self._process_input[model_name](model_name, batch, **kwargs)
            output = self._process_output[model_name](model_name, output)
            output.update(output)
        return output

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


__all__ = ["MultiSupervisedRunner"]
