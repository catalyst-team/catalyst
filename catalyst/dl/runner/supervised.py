from typing import Any, Callable, Dict, List, Mapping, Tuple, Union
from collections import OrderedDict
import logging
from pathlib import Path

import torch
from torch.jit import ScriptModule
from torch.utils.data import DataLoader

from catalyst.dl import (
    Callback,
    CheckpointCallback,
    InferCallback,
    State,
    SupervisedExperiment,
    utils,
)
from catalyst.utils.tools.typing import Device, Model

from .core import Runner

logger = logging.getLogger(__name__)


class SupervisedRunner(Runner):
    """Runner for experiments with supervised model."""

    _experiment_fn: Callable = SupervisedExperiment

    def __init__(
        self,
        model: Model = None,
        device: Device = None,
        input_key: Any = "features",
        output_key: Any = "logits",
        input_target_key: str = "targets",
    ):
        """
        Args:
            model (Module): Torch model object
            device (Device): Torch device
            input_key (Any): Key in batch dict mapping for model input
            output_key (Any): Key in output dict model output
                will be stored under
            input_target_key (str): Key in batch dict mapping for target
        """
        super().__init__(model=model, device=device)
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

    def _init(self):
        self.experiment: SupervisedExperiment = None
        self.state: State = None

    def _batch2device(self, batch: Mapping[str, Any], device: Device):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 2
            batch = {self.input_key: batch[0], self.target_key: batch[1]}
        batch = super()._batch2device(batch, device)
        return batch

    def _process_input_str(self, batch: Mapping[str, Any], **kwargs):
        output = self.model(batch[self.input_key], **kwargs)
        return output

    def _process_input_list(self, batch: Mapping[str, Any], **kwargs):
        input = {key: batch[key] for key in self.input_key}
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
        """
        output = self._process_input(batch, **kwargs)
        output = self._process_output(output)
        return output

    def _handle_batch(self, batch: Mapping[str, Any]) -> None:
        """
        Inner method to handle specified data batch.
        Used to make a train/valid/infer step during Experiment run.

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
        """
        self.state.batch_out = self.forward(batch)

    @torch.no_grad()
    def predict_batch(
        self, batch: Mapping[str, Any], **kwargs
    ) -> Mapping[str, Any]:
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
        batch = self._batch2device(batch, self.device)
        output = self.forward(batch, **kwargs)
        return output

    def infer(
        self,
        model: Model,
        loaders: "OrderedDict[str, DataLoader]",
        callbacks: "Union[List[Callback], OrderedDict[str, Callback]]" = None,
        verbose: bool = False,
        state_kwargs: Dict = None,
        fp16: Union[Dict, bool] = None,
        check: bool = False,
    ) -> None:
        """
        Makes the inference on the model.

        Args:
            model (Model): model to infer
            loaders (dict): dictionary containing one or several
                ``torch.utils.data.DataLoader`` for inference
            callbacks (List[catalyst.dl.Callback]): list of inference callbacks
            verbose (bool): ff true, it displays the status of the inference
                to the console.
            state_kwargs (dict): additional state params to ``State``
            fp16 (Union[Dict, bool]): If not None, then sets inference to FP16.
                See https://nvidia.github.io/apex/amp.html#properties
                if fp16=True, params by default will be ``{"opt_level": "O1"}``
            check (bool): if True, then only checks that pipeline is working
                (3 epochs only)
        """
        if isinstance(fp16, bool) and fp16:
            fp16 = {"opt_level": "O1"}

        experiment = self._experiment_fn(
            stage="infer",
            model=model,
            loaders=loaders,
            callbacks=callbacks,
            verbose=verbose,
            check_run=check,
            state_kwargs=state_kwargs,
            distributed_params=fp16,
        )
        self.run_experiment(experiment)

    def predict_loader(
        self,
        model: Model,
        loader: DataLoader,
        resume: str = None,
        verbose: bool = False,
        state_kwargs: Dict = None,
        fp16: Union[Dict, bool] = None,
        check: bool = False,
    ) -> Any:
        """
        Makes a prediction on the whole loader with the specified model.

        Args:
            model (Model): model to infer
            loader (DataLoader): dictionary containing only one
                ``torch.utils.data.DataLoader`` for inference
            resume (str): path to checkpoint for model
            verbose (bool): ff true, it displays the status of the inference
                to the console.
            state_kwargs (dict): additional state params to ``State``
            fp16 (Union[Dict, bool]): If not None, then sets inference to FP16.
                See https://nvidia.github.io/apex/amp.html#properties
                if fp16=True, params by default will be ``{"opt_level": "O1"}``
            check (bool): if True, then only checks that pipeline is working
                (3 epochs only)
        """
        loaders = OrderedDict([("infer", loader)])

        callbacks = OrderedDict([("inference", InferCallback())])
        if resume is not None:
            callbacks["loader"] = CheckpointCallback(resume=resume)

        self.infer(
            model=model,
            loaders=loaders,
            callbacks=callbacks,
            verbose=verbose,
            state_kwargs=state_kwargs,
            fp16=fp16,
            check=check,
        )

        output = callbacks["inference"].predictions
        if isinstance(self.output_key, str):
            output = output[self.output_key]

        return output

    def trace(
        self,
        model: Model = None,
        batch=None,
        logdir: str = None,
        loader: DataLoader = None,
        method_name: str = "forward",
        mode: str = "eval",
        requires_grad: bool = False,
        fp16: Union[Dict, bool] = None,
        device: Device = "cpu",
        predict_params: dict = None,
    ) -> ScriptModule:
        """
        Traces model using Torch Jit.

        Args:
            model (Model): model to trace
            batch: batch to forward through the model to trace
            logdir (str, optional): If specified,
                the result will be written to the directory
            loader (DataLoader, optional): if batch is not specified, the batch
                will be ``next(iter(loader))``
            method_name (str): model's method name that will be traced
            mode (str): ``train`` or ``eval``
            requires_grad (bool): flag to trace with gradients
            fp16 (Union[Dict, bool]): If not None, then sets
                tracing params to FP16
            device (Device): Torch deivice or a string
            predict_params (dict): additional parameters for model forward
        """
        if batch is None:
            if loader is None:
                raise ValueError(
                    "If batch is not provided the loader must be specified"
                )
            batch = next(iter(loader))

        if model is not None:
            self.model = model

        if isinstance(fp16, bool) and fp16:
            opt_level = "O1"
        elif isinstance(fp16, bool) and not fp16:
            opt_level = None
        elif isinstance(fp16, dict):
            opt_level = fp16["opt_level"]
        else:
            opt_level = fp16

        if opt_level is not None:
            device = "cuda"
        elif device is None:
            if self.device is None:
                self.device = utils.get_device()
            device = self.device

        result = utils.trace_model(
            model=self.model,
            runner=self,
            batch=batch,
            method_name=method_name,
            mode=mode,
            requires_grad=requires_grad,
            opt_level=opt_level,
            device=device,
            predict_params=predict_params,
        )

        if logdir is not None:
            filename = utils.get_trace_name(
                method_name=method_name,
                mode=mode,
                requires_grad=requires_grad,
                opt_level=opt_level,
            )

            logdir = Path(logdir)
            output: Path = logdir / "trace"
            output.mkdir(exist_ok=True, parents=True)

            out_model = str(output / filename)

            torch.jit.save(result, out_model)

        return result


__all__ = ["SupervisedRunner"]
