from typing import Callable, Dict, List  # isort:skip
import logging

from catalyst import utils
from catalyst.core import (
    _State, Callback, CallbackNode, CallbackOrder, registry
)
from catalyst.utils.tools.typing import Optimizer

logger = logging.getLogger(__name__)


class OptimizerCallback(Callback):
    """
    Optimizer callback, abstraction over optimizer step.
    """
    def __init__(
        self,
        loss_key: str = "loss",
        optimizer_key: str = None,
        accumulation_steps: int = 1,
        grad_clip_params: Dict = None,
        decouple_weight_decay: bool = True,
        # @TODO: add model grads support and visualization
        # save_model_grads: bool = False,
    ):
        """
        Args:
            grad_clip_params (dict): params for gradient clipping
            accumulation_steps (int): number of steps before
                ``model.zero_grad()``
            optimizer_key (str): A key to take a optimizer in case
                there are several of them and they are in a dictionary format.
            loss_key (str): key to get loss from ``state.loss``
            decouple_weight_decay (bool): If True - decouple weight decay
                regularization.
            # save_model_grads (bool): If True - State.model_grads will
            #     contain gradients calculated
            # on backward propagation on current
            #     batch
        """
        super().__init__(order=CallbackOrder.Optimizer, node=CallbackNode.All)
        self.loss_key: str = loss_key
        self.optimizer_key: str = optimizer_key

        self.accumulation_steps: int = accumulation_steps
        self._accumulation_counter: int = 0

        grad_clip_params: dict = grad_clip_params or {}
        self.grad_clip_fn = (
            registry.GRAD_CLIPPERS.get_from_params(**grad_clip_params)
        )

        self.decouple_weight_decay = decouple_weight_decay
        self._optimizer_wd: List[float] = [0.0]
        # self.save_model_grads = save_model_grads

    @staticmethod
    def grad_step(
        *,
        optimizer: Optimizer,
        optimizer_wds: List[float] = 0,
        grad_clip_fn: Callable = None
    ):
        """
        Makes a gradient step for a given optimizer

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
        optimizer.step()

    def on_stage_start(self, state: _State):
        """
        Checks that the current stage has correct optimizer
        """
        optimizer = state.get_attr(
            key="optimizer", inner_key=self.optimizer_key
        )
        assert optimizer is not None
        self._optimizer = optimizer

    def on_epoch_start(self, state: _State):
        """On epoch start event"""
        optimizer = self._optimizer

        if self.decouple_weight_decay:
            self._optimizer_wd = [
                group.get("weight_decay", 0.0)
                for group in optimizer.param_groups
            ]
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["weight_decay"] = 0.0
        else:
            self._optimizer_wd = [0.0] * len(optimizer.param_groups)

    def on_epoch_end(self, state: _State):
        """On epoch end event"""
        if self.decouple_weight_decay:
            optimizer = self._optimizer
            for i, wd in enumerate(self._optimizer_wd):
                optimizer.param_groups[i]["weight_decay"] = wd

    def on_batch_end(self, state: _State):
        """On batch end event"""
        if not state.need_backward_pass:
            return

        loss = state.batch_metrics[self.loss_key]
        optimizer = self._optimizer

        self._accumulation_counter += 1
        need_gradient_step = \
            (self._accumulation_counter + 1) % self.accumulation_steps == 0

        # This is very hacky check whether we have AMP optimizer and this may
        # change in future.
        # But alternative solution is to have AmpOptimizerCallback.
        # or expose another c'tor argument.
        if hasattr(optimizer, "_amp_stash"):
            from apex import amp
            # Need to set ``delay_unscale``
            # according to
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
            delay_unscale = not need_gradient_step
            with amp.scale_loss(
                loss, optimizer, delay_unscale=delay_unscale
            ) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if need_gradient_step:
            self.grad_step(
                optimizer=optimizer,
                optimizer_wds=self._optimizer_wd,
                grad_clip_fn=self.grad_clip_fn
            )

            # if self.save_model_grads:
            #     for tag, value in model.named_parameters():
            #         tag = tag.replace(".", "/")
            #         state.model_grads[tag] = value.grad.cpu().numpy()

            utils.maybe_recursive_call(optimizer, "zero_grad")

            self._accumulation_counter = 0


__all__ = ["OptimizerCallback"]
