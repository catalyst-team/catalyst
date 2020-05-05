from typing import Union  # isort:skip
from operator import lt, gt


from pathlib import Path

from catalyst.dl import Callback, CallbackOrder, State
from catalyst.dl.utils import (
    trace_model_from_state,
    save_traced_model,
)


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
        :param trace_mode (str): Mode for model to trace
            (``train`` or ``eval``)
        :param requires_grad (bool): Flag to use grads
        :param opt_level (str): AMP FP16 init level
        :param out_dir (str): Directory to save model to
        :param out_model: Path to save model to (override out_dir)
        """

        if trace_mode not in ["train", "eval"]:
            raise ValueError(
                f"Unknown trace_mode '{trace_mode}'. "
                f"Must be 'eval' or 'train'")

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

            traced_model = trace_model_from_state(
                state=state,
                method_name=self.method_name,
                loader=self.loader,
                mode=self.trace_mode,
                requires_grad=self.requires_grad,
                opt_level=self.opt_level,
                device=device,
            )

            save_traced_model(
                model=traced_model,
                logdir=state.logdir,
                method_name=self.method_name,
                mode=self.trace_mode,
                requires_grad=self.requires_grad,
                opt_level=self.opt_level,
                out_model=self.out_model,
                out_dir=self.out_dir,
            )


__all__ = [
    "TracerCallback"
]
