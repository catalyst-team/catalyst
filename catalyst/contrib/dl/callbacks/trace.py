from typing import Any, Dict, List, Optional, Callable, Union  # isort:skip
from operator import lt, gt

import inspect
import torch

from torch.jit import ScriptModule
from pathlib import Path

from catalyst.dl import Callback, CallbackOrder, State
from catalyst.utils.tools.typing import Device, Model
from catalyst.dl.utils import (get_fn_argsnames, any2device,
                               get_trace_name, trace_model)


def _get_native_batch(
        loaders, loader: Union[str, int] = 0, data_index: int = 0
):
    """Returns a batch from experiment loader

    Args:
        loaders (Dict[DataLoader]): loaders
        loader (Union[str, int]): loader name or its index,
            default is the first loader
        data_index (int): index in dataset from the loader
    """
    if isinstance(loader, str):
        _loader = loaders[loader]
    elif isinstance(loader, int):
        _loader = list(loaders.values())[loader]
    else:
        raise TypeError("Loader parameter must be a string or an integer")

    dataset = _loader.dataset
    collate_fn = _loader.collate_fn

    sample = collate_fn([dataset[data_index]])

    return sample


# almost the same as catalyst.scripts.trace.trace_model_from_checkpoint
def trace_model_from_state(
    state: State,
    method_name: str = "forward",
    loader: Union[str, int] = None,
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    device: Device = "cpu",
):
    """
    Traces model using created experiment and runner.

    Args:
        state (State): Current runner state.
        method_name (str): Model's method name that will be
            used as entrypoint during tracing
        loader (Union[str, int]): experiment's loader name or its index
        mode (str): Mode for model to trace (``train`` or ``eval``)
        requires_grad (bool): Flag to use grads
        opt_level (str): AMP FP16 init level
        device (str): Torch device

    Returns:
        the traced model
    """

    if loader is None:
        loader = 0

    _model = state.model
    if isinstance(_model, (torch.nn.DataParallel,
                           torch.nn.parallel.DistributedDataParallel)):
        _model = _model.module

    batch = _get_native_batch(state.loaders, loader)

    # getting input names of args for method since we don't have Runner
    # and we don't know input_key to preprocess batch for method call
    fn = getattr(_model, method_name)
    argspec = inspect.getfullargspec(fn)
    assert (
        argspec.varargs is None and argspec.varkw is None
    ), "not supported by PyTorch tracing"
    method_argnames = get_fn_argsnames(fn, exclude=["self"])

    batch = {name: batch[name] for name in method_argnames}
    batch = any2device(batch, device)

    # Dumping previous state of the model, we will need it to restore
    _device = state.device
    _is_training = _model.training
    _requires_grads = {}
    for name, param in _model.named_parameters():
        _requires_grads[name] = param.requires_grad

    # function to run prediction on batch
    def predict_fn(model: Model, inputs, **kwargs):
        return model(**inputs, **kwargs)

    _model.to(device)

    print("Tracing")
    traced = trace_model(
        model=_model,
        predict_fn=predict_fn,
        batch=batch,
        method_name=method_name,
        mode=mode,
        requires_grad=requires_grad,
        opt_level=opt_level,
        device=device,
    )

    # Restore previous state of the model, otherwise training will crush
    for name, param in _model.named_parameters():
        param.requires_grad = _requires_grads[name]

    _model.to(_device)
    _model.train(mode=_is_training)

    print("Done")
    return traced


class TracerCallback(Callback):

    def __init__(
            self,
            metric_key: str,
            stage: str = None,
            mode: str = "max",
            method_name: str = "forward",
            requires_grad: bool = False,
            opt_level: str = None,
            loader: Union[str, int] = None,
            trace_mode: str = "eval",
            out_dir: Union[str, Path] = None,
            out_model: Union[str, Path] = None,
    ):
        """

        Traces model using created experiment and runner.

        :param metric_key (str): Metric key we should trace model based on
        :param mode (str): Metric max or min value affects tracing.
        :param method_name (str): Model's method name that will be
            used as entrypoint during tracing
        :param checkpoint_name (str): Checkpoint's name to trace
        :param stage (str): Stage from experiment from which model and loader
            will be taken
        :param loader (str): Loader name to get the batch from
        :param trace_mode (str): Mode for model to trace (``train`` or ``eval``)
        :param requires_grad (bool): Flag to use grads
        :param opt_level:
        :param out_dir:
        :param out_model:
        """

        if trace_mode not in ["train", "eval"]:
            raise ValueError(
                f"Unknown trace_mode '{trace_mode}'. Must be 'eval' or 'train'")

        if mode == "max":
            self.compare_fn = gt
            self.default_value = float('-inf')
        elif mode == "min":
            self.compare_fn = lt
            self.default_value = float('inf')
        else:
            raise ValueError(f"Unknown mode '{mode}. Must be 'max' or 'min'")

        self.metric_key = metric_key

        self.requires_grad = requires_grad
        self.method_name = method_name
        self.trace_mode = trace_mode
        self.opt_level = opt_level
        self.stage = stage

        if out_model is not None and not isinstance(out_model, Path):
            out_model = Path(out_model)

        if out_dir is not None and not isinstance(out_model, Path):
            out_dir = Path(out_dir)

        self.out_model = out_model
        self.out_dir = out_dir

        if loader is None:
            loader = 0
        self.loader = loader

        self.value = self.default_value

        super(TracerCallback, self).__init__(CallbackOrder.External)

    def on_epoch_start(self, state: State):
        self.value = self.default_value

    def on_epoch_end(self, state: State):

        if self.stage is not None and state.stage_name != self.stage:
            return

        value = state.valid_metrics[self.metric_key]

        if self.compare_fn(value, self.value):

            self.value = value

            if self.opt_level is not None:
                device = "cuda"
            else:
                device = "cpu"

            traced = trace_model_from_state(
                state=state,
                method_name=self.method_name,
                loader=self.loader,
                mode=self.trace_mode,
                requires_grad=self.requires_grad,
                opt_level=self.opt_level,
                device=device,
            )

            if self.out_model is None:
                file_name = get_trace_name(
                    method_name=self.method_name,
                    mode=self.trace_mode,
                    requires_grad=self.requires_grad,
                    opt_level=self.opt_level,
                    additional_string=None,
                )

                output: Path = self.out_dir
                if output is None:
                    output: Path = state.logdir / "trace"
                output.mkdir(exist_ok=True, parents=True)

                out_model = str(output / file_name)
            else:
                out_model = str(self.out_model)

            torch.jit.save(traced, out_model)
