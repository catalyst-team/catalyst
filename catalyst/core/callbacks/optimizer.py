from typing import Callable, Dict, List
import logging
import warnings

import torch

from catalyst import registry
from catalyst.core import utils
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.tools.typing import Optimizer

logger = logging.getLogger(__name__)

try:
    import torch_xla.core.xla_model as xm
except ModuleNotFoundError:
    pass


def zero_grad(optimizer: Optimizer) -> None:
    """Perform an hacky way to zero gradients.

    Args:
        optimizer (Optimizer): optimizer with model parameters.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = None


class IOptimizerCallback(Callback):
    """Optimizer callback interface, abstraction over optimizer step."""

    pass


class OptimizerCallback(IOptimizerCallback):
    """Optimizer callback, abstraction over optimizer step."""

    def __init__(
        self,
        metric_key: str = None,
        optimizer_key: str = None,
        accumulation_steps: int = 1,
        grad_clip_params: Dict = None,
        decouple_weight_decay: bool = True,
        loss_key: str = None,
        use_fast_zero_grad: bool = False,
        xla_barrier: bool = True,
    ):
        """
        Args:
            loss_key (str): key to get loss from ``runner.batch_metrics``
            optimizer_key (str): A key to take a optimizer in case
                there are several of them and they are in a dictionary format.
            accumulation_steps (int): number of steps before
                ``model.zero_grad()``
            grad_clip_params (dict): params for gradient clipping
            decouple_weight_decay (bool): If ``True`` - decouple weight decay
                regularization.
            use_fast_zero_grad (bool): boost ``optiomizer.zero_grad()``,
                default is ``False``.
            xla_barrier (bool): barrier option for xla. Here you can find
                more about usage of `barrier flag
                <https://pytorch.org/xla/release/1.5/index.html?
                highlight=optimizer_step#torch_xla.core.xla_model.optimizer_step>`_
                and `examples
                <https://pytorch.org/xla/release/1.5/index.html#
                running-on-a-single-xla-device>`_.

                Default is ``True``.
        """
        super().__init__(order=CallbackOrder.optimizer, node=CallbackNode.all)
        assert metric_key is None or loss_key is None
        if loss_key is not None:
            warnings.warn(
                "OptimizerCallback: "
                "`loss_key` is now deprecated in favor `metric_key`",
                stacklevel=2,
            )
        self.metric_key: str = metric_key or loss_key or "loss"
        self.optimizer_key: str = optimizer_key

        self.accumulation_steps: int = accumulation_steps
        self._accumulation_counter: int = 0

        grad_clip_params: dict = grad_clip_params or {}
        self.grad_clip_fn = registry.GRAD_CLIPPER.get_from_params(
            **grad_clip_params
        )

        self.decouple_weight_decay = decouple_weight_decay
        self._optimizer_wd: List[float] = [0.0]
        self._optimizer_step_fn: Callable = None
        self.is_xla = False
        self.use_fast_zero_grad = use_fast_zero_grad
        self.use_xla_barrier = xla_barrier

    def _optimizer_step(self, optimizer: Optimizer) -> None:
        """CPU and GPU optimization step.

        Args:
            optimizer (Optimizer): optimizer object
        """
        optimizer.step()

    def _optimizer_step_tpu(self, optimizer: Optimizer) -> None:
        """TPU optimization step.

        Args:
            optimizer (Optimizer): optimizer object
        """
        if self.use_xla_barrier:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            xm.optimizer_step(optimizer)

    def grad_step(
        self,
        *,
        optimizer: Optimizer,
        optimizer_wds: List[float] = 0,
        grad_clip_fn: Callable = None,
    ) -> None:
        """Makes a gradient step for a given optimizer.

        Args:
            optimizer (Optimizer): the optimizer
            optimizer_wds (List[float]): list of weight decay parameters
                for each param group
            grad_clip_fn (Callable): function for gradient clipping
        """
        for group, wd in zip(optimizer.param_groups, optimizer_wds):
            if wd > 0:
                for param in group["params"]:
                    param.data = param.data.add(-wd * group["lr"], param.data)
            if grad_clip_fn is not None:
                grad_clip_fn(group["params"])
        # optimize parameters
        self._optimizer_step_fn(optimizer)

    def on_stage_start(self, runner: IRunner) -> None:
        """Checks that the current stage has correct optimizer.

        Args:
            runner(IRunner): current runner
        """
        self._optimizer = runner.get_attr(
            key="optimizer", inner_key=self.optimizer_key
        )
        # device based optimization step
        if runner.device.type == "xla":
            self._optimizer_step_fn = self._optimizer_step_tpu
        else:
            self._optimizer_step_fn = self._optimizer_step

        assert self._optimizer is not None

    def on_epoch_start(self, runner: IRunner) -> None:
        """On epoch start event.

        Args:
            runner (IRunner): current runner
        """
        if self.decouple_weight_decay:
            self._optimizer_wd = [
                group.get("weight_decay", 0.0)
                for group in self._optimizer.param_groups
            ]
            for i in range(len(self._optimizer.param_groups)):
                self._optimizer.param_groups[i]["weight_decay"] = 0.0
        else:
            self._optimizer_wd = [0.0] * len(self._optimizer.param_groups)

    def on_batch_end(self, runner: IRunner) -> None:
        """On batch end event

        Args:
            runner (IRunner): current runner
        """
        if not runner.is_train_loader:
            return

        loss = runner.batch_metrics[self.metric_key]

        self._accumulation_counter += 1
        need_gradient_step = (
            self._accumulation_counter % self.accumulation_steps == 0
        )

        # This is very hacky check whether we have AMP optimizer and this may
        # change in future.
        # But alternative solution is to have AmpOptimizerCallback.
        # or expose another c'tor argument.
        # @TODO: speedup with re-definition ``on_stage_start``
        if hasattr(self._optimizer, "_amp_stash"):
            from apex import amp

            # Need to set ``delay_unscale``
            # according to
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
            delay_unscale = not need_gradient_step
            with amp.scale_loss(
                loss, self._optimizer, delay_unscale=delay_unscale
            ) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if need_gradient_step:
            self.grad_step(
                optimizer=self._optimizer,
                optimizer_wds=self._optimizer_wd,
                grad_clip_fn=self.grad_clip_fn,
            )
            if not self.use_fast_zero_grad:
                utils.maybe_recursive_call(self._optimizer, "zero_grad")
            else:
                utils.maybe_recursive_call(self._optimizer, zero_grad)
            self._accumulation_counter = 0

    def on_epoch_end(self, runner: IRunner) -> None:
        """On epoch end event.

        Args:
            runner (IRunner): current runner
        """
        if self.decouple_weight_decay:
            for i, wd in enumerate(self._optimizer_wd):
                self._optimizer.param_groups[i]["weight_decay"] = wd

        lr = self._optimizer.param_groups[0]["lr"]
        lr_name = (
            f"lr/{self.optimizer_key}"
            if self.optimizer_key is not None
            else "lr"
        )
        runner.epoch_metrics[lr_name] = lr

        momentum = utils.get_optimizer_momentum(self._optimizer)
        if momentum is not None:
            momentum_name = (
                f"momentum/{self.optimizer_key}"
                if self.optimizer_key is not None
                else "momentum"
            )
            runner.epoch_metrics[momentum_name] = momentum


class AMPOptimizerCallback(IOptimizerCallback):
    """
    Optimizer callback with native torch amp support.
    """

    def __init__(
        self,
        metric_key: str = None,
        optimizer_key: str = None,
        accumulation_steps: int = 1,
        grad_clip_params: Dict = None,
        loss_key: str = None,
    ):
        """
        Args:
            loss_key (str): key to get loss from ``runner.batch_metrics``
            optimizer_key (str): A key to take a optimizer in case
                there are several of them and they are in a dictionary format.
            accumulation_steps (int): number of steps before
                ``model.zero_grad()``
            grad_clip_params (dict): params for gradient clipping
            decouple_weight_decay (bool): If True - decouple weight decay
                regularization.
        """
        super().__init__(order=CallbackOrder.optimizer, node=CallbackNode.all)
        assert metric_key is None or loss_key is None
        if loss_key is not None:
            warnings.warn(
                "OptimizerCallback: "
                "`loss_key` is now deprecated in favor `metric_key`",
                stacklevel=2,
            )
        self.metric_key: str = metric_key or loss_key or "loss"
        self.optimizer_key: str = optimizer_key

        self.accumulation_steps: int = accumulation_steps
        self._accumulation_counter: int = 0

        grad_clip_params: dict = grad_clip_params or {}
        self.grad_clip_fn = registry.GRAD_CLIPPER.get_from_params(
            **grad_clip_params
        )

        # Initialized at on_state_start()
        self.scaler = None

    def grad_step(
        self, *, optimizer: Optimizer, grad_clip_fn: Callable = None,
    ) -> None:
        """Makes a gradient step for a given optimizer.

        Args:
            optimizer (Optimizer): the optimizer
            grad_clip_fn (Callable): function for gradient clipping
        """
        if grad_clip_fn is not None:
            # Unscales the gradients of
            # optimizer's assigned params in-place
            self.scaler.unscale_(optimizer)
            for group in zip(optimizer.param_groups):
                # Since the gradients of optimizer's
                # assigned params are unscaled, clips as usual:
                grad_clip_fn(group["params"])

        self.scaler.step(optimizer)
        self.scaler.update()

    def on_stage_start(self, runner: IRunner) -> None:
        """Checks that the current stage has correct optimizer.

        Args:
            runner(IRunner): current runner
        """
        from torch.cuda.amp import GradScaler

        self._optimizer = runner.get_attr(
            key="optimizer", inner_key=self.optimizer_key
        )
        self.scaler = GradScaler()
        assert self._optimizer is not None

    def on_batch_start(self, runner: IRunner) -> None:
        """On batch start event

        Args:
            runner (IRunner): current runner
        """
        self.prev_autocast_state = torch.is_autocast_enabled()
        torch.set_autocast_enabled(True)
        torch.autocast_increment_nesting()

    def on_batch_end(self, runner: IRunner) -> None:
        """On batch end event

        Args:
            runner (IRunner): current runner
        """
        # Drop the cache when we exit to a nesting level
        # that's outside any instance of autocast.
        if torch.autocast_decrement_nesting() == 0:
            torch.clear_autocast_cache()
        torch.set_autocast_enabled(self.prev_autocast_state)

        if not runner.is_train_loader:
            return

        loss = runner.batch_metrics[self.metric_key]

        self._accumulation_counter += 1
        need_gradient_step = (
            self._accumulation_counter % self.accumulation_steps == 0
        )

        self.scaler.scale(loss).backward()

        if need_gradient_step:
            self.grad_step(
                optimizer=self._optimizer, grad_clip_fn=self.grad_clip_fn,
            )

            utils.maybe_recursive_call(self._optimizer, "zero_grad")
            self._accumulation_counter = 0

    def on_epoch_end(self, runner: IRunner) -> None:
        """On epoch end event.

        Args:
            runner (IRunner): current runner
        """
        lr = self._optimizer.param_groups[0]["lr"]
        lr_name = (
            f"lr/{self.optimizer_key}"
            if self.optimizer_key is not None
            else "lr"
        )
        runner.epoch_metrics[lr_name] = lr

        momentum = utils.get_optimizer_momentum(self._optimizer)
        if momentum is not None:
            momentum_name = (
                f"momentum/{self.optimizer_key}"
                if self.optimizer_key is not None
                else "momentum"
            )
            runner.epoch_metrics[momentum_name] = momentum

    def on_stage_end(self, runner: IRunner) -> None:
        """On stage end event.

        Args:
            runner (IRunner): current runner
        """
        self.scaler = None


# @TODO: add OptimizerCallback autocreation
# def OptimizerCallback(*args, **kwargs):
#     """
#     Optimizer callback factory-wrapper to select required OptimizerCallback
#     automatically.
#     """
#     is_amp_enabled = (
#         os.getenv("USE_AMP", "0") == "1" and utils.check_amp_available()
#     )
#
#     optimizer_callback = AMPOptimizerCallback(*args, **kwargs) \
#         if is_amp_enabled \
#         else OptimizerCallback(*args, **kwargs)
#     return optimizer_callback


__all__ = [
    "IOptimizerCallback",
    "AMPOptimizerCallback",
    "OptimizerCallback",
]
