from typing import Callable, Dict, List, TYPE_CHECKING
import logging
import warnings

import torch

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.registry import REGISTRY
from catalyst.typing import Optimizer
from catalyst.utils.misc import get_attr, maybe_recursive_call
from catalyst.utils.torch import get_optimizer_momentum

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner

logger = logging.getLogger(__name__)

try:
    import torch_xla.core.xla_model as xm
except ModuleNotFoundError:
    pass


def zero_grad(optimizer: Optimizer) -> None:
    """Perform an hacky way to zero gradients.

    Args:
        optimizer: optimizer with model parameters.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = None


class IOptimizerCallback(Callback):
    """Optimizer callback interface, abstraction over optimizer step."""

    pass


# class OptimizerCallback(IOptimizerCallback):
#     """Optimizer callback, abstraction over optimizer step."""
#
#     def __init__(
#         self,
#         metric_key: str = None,
#         optimizer_key: str = None,
#         accumulation_steps: int = 1,
#         grad_clip_params: Dict = None,
#         decouple_weight_decay: bool = False,
#         use_fast_zero_grad: bool = False,
#         xla_barrier: bool = True,
#         use_amp: bool = None,
#         use_apex: bool = None,
#     ):
#         """
#         Args:
#             loss_key: key to get loss from ``runner.batch_metrics``
#             optimizer_key: A key to take a optimizer in case
#                 there are several of them and they are in a dictionary format.
#             accumulation_steps: number of steps before ``model.zero_grad()``
#             grad_clip_params: params for gradient clipping, example:
#                 ``{'func': 'clip_grad_norm_', 'max_norm': 1, norm_type': 2}``
#             decouple_weight_decay: If ``True`` - decouple weight decay
#                 regularization, default is ``False``.
#             use_fast_zero_grad: boost ``optimizer.zero_grad()``,
#                 default is ``False``.
#             xla_barrier: barrier option for xla. Here you can find
#                 more about usage of `barrier flag
#                 <https://pytorch.org/xla/release/1.5/index.html?
#                 highlight=optimizer_step#torch_xla.core.xla_model.optimizer_step>`_
#                 and `examples
#                 <https://pytorch.org/xla/release/1.5/index.html#
#                 running-on-a-single-xla-device>`_.
#                 Default is ``True``.
#             use_amp: whether to use native pytorch AMP, if None will be set
#                 based on runner.experiment.engine_params on stage start
#             use_apex: whether to use apex, if None will be set
#                 based on runner.experiment.engine_params on stage start
#
#         """
#         super().__init__(order=CallbackOrder.optimizer, node=CallbackNode.all)
#         self.metric_key: str = metric_key or "loss"
#         self.optimizer_key: str = optimizer_key
#
#         self.accumulation_steps: int = accumulation_steps
#         self._accumulation_counter: int = 0
#
#         if use_apex and use_amp:
#             raise ValueError(
#                 "OptimizerCallback: ``use_amp==True`` and ``use_apex==True`` "
#                 "You must choose only one mixed precision backend"
#             )
#
#         self.use_amp = use_amp
#         self.use_apex = use_apex
#
#         # If use_amp==True scaler is initialized at on_stage_start()
#         self.scaler = None
#
#         grad_clip_params: dict = grad_clip_params or {}
#         self.grad_clip_fn = REGISTRY.get_from_params(**grad_clip_params)
#
#         self.decouple_weight_decay = decouple_weight_decay
#         self._optimizer_wds: List = None
#         self._optimizer_step_fn: Callable = None
#         self.is_xla = False
#         self.use_fast_zero_grad = use_fast_zero_grad
#         self.use_xla_barrier = xla_barrier
#
#     def _optimizer_step(self) -> None:
#         """CPU and GPU optimization step.
#         """
#         self._optimizer.step()
#
#     def _optimizer_step_amp(self) -> None:
#         """Optimization step with pytorch native amp
#         """
#         self.scaler.step(self._optimizer)
#         self.scaler.update()
#
#     def _optimizer_step_tpu(self) -> None:
#         """TPU optimization step.
#         """
#         if self.use_xla_barrier:
#             xm.optimizer_step(self._optimizer, barrier=True)
#         else:
#             xm.optimizer_step(self._optimizer)
#
#     def grad_step(
#         self, *, optimizer: Optimizer, grad_clip_fn: Callable = None
#     ) -> None:
#         """Makes a gradient step for a given optimizer.
#
#         Args:
#             optimizer: the optimizer
#             grad_clip_fn: function for gradient clipping
#         """
#         if self.decouple_weight_decay:
#             for group, wd in zip(optimizer.param_groups, self._optimizer_wds):
#                 for param in group["params"]:
#                     param.data = param.data.add(
#                         other=param.data, alpha=-wd * group["lr"]
#                     )
#
#         if grad_clip_fn is not None:
#             if self.use_amp:
#                 self.scaler.unscale_(optimizer)
#             for group in optimizer.param_groups:
#                 grad_clip_fn(group["params"])
#
#         # optimize parameters
#         self._optimizer_step_fn(optimizer)
#
#     def on_stage_start(self, runner: "IRunner") -> None:
#         """Resolve amp/apex settings, prepare optimizer and scaler
#
#         Args:
#             runner(IRunner): current runner
#         """
#         if self.use_amp is None:
#             if runner.experiment is not None:
#                 self.use_amp = runner.experiment.engine_params.get(
#                     "amp", False
#                 )
#             else:
#                 self.use_amp = False
#
#         if self.use_apex is None:
#             if runner.experiment is not None:
#                 self.use_apex = runner.experiment.engine_params.get(
#                     "apex", False
#                 )
#             else:
#                 self.use_apex = False
#
#         self._optimizer = get_attr(
#             runner, key="optimizer", inner_key=self.optimizer_key
#         )
#         self._optimizer_step_fn = runner.experiment.engine.optimizer_step
#         # # device based optimization step
#         # if runner.device.type == "xla":
#         #     self._optimizer_step_fn = self._optimizer_step_tpu
#         # elif self.use_amp:
#         #     self._optimizer_step_fn = self._optimizer_step_amp
#         # else:
#         #     self._optimizer_step_fn = self._optimizer_step
#
#         if hasattr(self._optimizer, "_amp_stash") and not self.use_apex:
#             warnings.warn(
#                 "`_amp_stash` is found in `self._optimizer`:, "
#                 "but `use_apex` is False",
#                 stacklevel=2,
#             )
#
#         assert self._optimizer is not None
#
#         if self.use_amp:
#             from torch.cuda.amp import GradScaler
#
#             self.scaler = GradScaler()
#
#     def on_epoch_start(self, runner: "IRunner") -> None:
#         """On epoch start event.
#
#         Args:
#             runner: current runner
#         """
#         if self.decouple_weight_decay:
#             self._optimizer_wds = [
#                 group.get("weight_decay", 0.0)
#                 for group in self._optimizer.param_groups
#             ]
#             for i in range(len(self._optimizer.param_groups)):
#                 self._optimizer.param_groups[i]["weight_decay"] = 0.0
#
#     def on_batch_start(self, runner: "IRunner") -> None:
#         """On batch start event
#
#         Args:
#             runner: current runner
#         """
#         if self.use_amp:
#             self.prev_autocast_state = torch.is_autocast_enabled()
#             torch.set_autocast_enabled(True)
#             torch.autocast_increment_nesting()
#
#     def on_batch_end(self, runner: "IRunner") -> None:
#         """On batch end event
#
#         Args:
#             runner: current runner
#         """
#         if self.use_amp:
#             # Drop the cache when we exit to a nesting level
#             # that's outside any instance of autocast.
#             if torch.autocast_decrement_nesting() == 0:
#                 torch.clear_autocast_cache()
#             torch.set_autocast_enabled(self.prev_autocast_state)
#
#         if not runner.is_train_loader:
#             return
#
#         loss = runner.batch_metrics[self.metric_key]
#
#         self._accumulation_counter += 1
#         need_gradient_step = (
#             self._accumulation_counter % self.accumulation_steps == 0
#         )
#
#         # @TODO: speedup with re-definition ``on_stage_start``
#         if self.use_apex:
#             from apex import amp
#
#             # Need to set ``delay_unscale``
#             # according to
#             # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
#             delay_unscale = not need_gradient_step
#             with amp.scale_loss(
#                 loss, self._optimizer, delay_unscale=delay_unscale
#             ) as scaled_loss:
#                 scaled_loss.backward()
#         elif self.use_amp:
#             self.scaler.scale(loss).backward()
#         else:
#             loss.backward()
#
#         if need_gradient_step:
#             self.grad_step(
#                 optimizer=self._optimizer, grad_clip_fn=self.grad_clip_fn,
#             )
#             if not self.use_fast_zero_grad:
#                 maybe_recursive_call(self._optimizer, "zero_grad")
#             else:
#                 maybe_recursive_call(self._optimizer, zero_grad)
#             self._accumulation_counter = 0
#
#     def on_epoch_end(self, runner: "IRunner") -> None:
#         """On epoch end event.
#
#         Args:
#             runner: current runner
#         """
#         if self.decouple_weight_decay:
#             for i, wd in enumerate(self._optimizer_wds):
#                 self._optimizer.param_groups[i]["weight_decay"] = wd
#
#         lr = self._optimizer.param_groups[0]["lr"]
#         lr_name = (
#             f"lr/{self.optimizer_key}"
#             if self.optimizer_key is not None
#             else "lr"
#         )
#         runner.epoch_metrics[lr_name] = lr
#
#         momentum = get_optimizer_momentum(self._optimizer)
#         if momentum is not None:
#             momentum_name = (
#                 f"momentum/{self.optimizer_key}"
#                 if self.optimizer_key is not None
#                 else "momentum"
#             )
#             runner.epoch_metrics[momentum_name] = momentum
#
#     def on_stage_end(self, runner: "IRunner") -> None:
#         """On stage end event.
#
#         Args:
#             runner: current runner
#         """
#         if self.use_amp:
#             self.scaler = None


class OptimizerCallback(IOptimizerCallback):
    def __init__(
        self,
        metric_key: str = None,
        model_key: str = None,
        optimizer_key: str = None,
        criterion_key: str = None,
        accumulation_steps: int = 1,
        grad_clip_params: Dict = None,
    ):
        super().__init__(order=CallbackOrder.optimizer, node=CallbackNode.all)
        self.metric_key = metric_key
        self.model_key = model_key
        self.optimizer_key = optimizer_key
        self.criterion_key = criterion_key
        self.model = None
        self.optimizer = None
        self.criterion = None

    def on_stage_start(self, runner: "IRunner") -> None:
        self.model = get_attr(runner, key="model", inner_key=self.model_key)
        self.optimizer = get_attr(runner, key="optimizer", inner_key=self.optimizer_key)
        self.criterion = get_attr(runner, key="criterion", inner_key=self.criterion_key)
        assert self.model is not None
        assert self.optimizer is not None

    def on_batch_end(self, runner: "IRunner"):
        if runner.is_train_loader:
            loss = runner.batch_metrics[self.metric_key]
            # @TODO: do we need criterion here? Looks like no :)
            runner.engine.zero_grad(self.model, self.criterion, self.optimizer, loss)
            runner.engine.backward_loss(self.model, self.criterion, self.optimizer, loss)
            runner.engine.optimizer_step(self.model, self.criterion, self.optimizer, loss)

            # runner.batch_metrics.update({"lr": runner.optimizer.lr})

    # def on_loader_end(self, runner: "IRunner") -> None:
    #     runner.loader_metrics.update({"lr": runner.optimizer.lr})


__all__ = [
    "IOptimizerCallback",
    "OptimizerCallback",
]
