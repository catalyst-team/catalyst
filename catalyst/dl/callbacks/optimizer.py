import logging
from typing import Dict, List, Callable

import safitty
import torch

from catalyst.dl.core import Callback, RunnerState, CallbackOrder
from catalyst.dl.registry import GRAD_CLIPPERS
from catalyst.dl.utils import get_optimizer_momentum, maybe_recursive_call
from catalyst.dl.utils.torch import _Optimizer

logger = logging.getLogger(__name__)


class OptimizerCallback(Callback):
    """
    Optimizer callback, abstraction over optimizer step.
    """

    def __init__(
        self,
        grad_clip_params: Dict = None,
        accumulation_steps: int = 1,
        optimizer_key: str = None,
        loss_key: str = "loss",
        decouple_weight_decay: bool = True
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
        """
        super().__init__(CallbackOrder.Optimizer)
        grad_clip_params: dict = grad_clip_params or {}
        self.grad_clip_fn = GRAD_CLIPPERS.get_from_params(**grad_clip_params)

        self.accumulation_steps: int = accumulation_steps
        self.optimizer_key: str = optimizer_key
        self.loss_key: str = loss_key
        self.decouple_weight_decay = decouple_weight_decay
        self._optimizer_wd: List[float] = [0.0]
        self._accumulation_counter: int = 0

    @staticmethod
    def grad_step(
        *,
        optimizer: _Optimizer,
        optimizer_wds: List[float] = 0,
        grad_clip_fn: Callable = None
    ):
        for group, wd in zip(optimizer.param_groups, optimizer_wds):
            if wd > 0:
                for param in group["params"]:
                    param.data = param.data.add(
                        -wd * group["lr"], param.data
                    )
            if grad_clip_fn is not None:
                grad_clip_fn(group["params"])
        optimizer.step()

    def on_stage_start(self, state: RunnerState):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        assert optimizer is not None
        lr = optimizer.defaults["lr"]
        momentum = get_optimizer_momentum(optimizer)
        state.set_key(lr, "lr", inner_key=self.optimizer_key)
        state.set_key(momentum, "momentum", inner_key=self.optimizer_key)

    def on_epoch_start(self, state):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        if self.decouple_weight_decay:
            self._optimizer_wd = [
                group.get("weight_decay", 0.0)
                for group in optimizer.param_groups
            ]
            for i in range(len(optimizer.param_groups)):
                safitty.set(
                    optimizer.param_groups, i, "weight_decay", value=0.0)
        else:
            self._optimizer_wd = [0.0] * len(optimizer.param_groups)

    def _get_loss(self, state) -> torch.Tensor:
        loss = state.get_key(key="loss", inner_key=self.loss_key)

        if isinstance(loss, list):
            raise ValueError(
                f"Loss is a list. "
                f"Only the last value will be used for `backward`."
                f"To aggregate losses into "
                "one value use `CriterionAggregatorCallback`"
            )
        if isinstance(loss, dict):
            error = f"Loss is a dict: {list(loss.keys())}, " \
                    f"to aggregate losses into " \
                    "one value use `CriterionAggregatorCallback`."
            if self.loss_key is None:
                error = error + " Or try to pass `loss_key` " \
                                "in the OptimizerCallback init"
            raise ValueError(error)
        return loss

    def on_batch_start(self, state):
        state.loss = None

    def on_batch_end(self, state):
        if not state.need_backward:
            return

        loss = self._get_loss(state)

        self._accumulation_counter += 1
        model = state.model
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )

        # This is very hacky check whether we have AMP optimizer and this may
        # change in future.
        # But alternative solution is to have AmpOptimizerCallback.
        # or expose another c'tor argument.
        if hasattr(optimizer, "_amp_stash"):
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (self._accumulation_counter + 1) % self.accumulation_steps == 0:
            self.grad_step(
                optimizer=optimizer,
                optimizer_wds=self._optimizer_wd,
                grad_clip_fn=self.grad_clip_fn
            )
            maybe_recursive_call(model, "zero_grad")

            self._accumulation_counter = 0

    def on_epoch_end(self, state):
        if self.decouple_weight_decay:
            optimizer = state.get_key(
                key="optimizer", inner_key=self.optimizer_key
            )
            for i, wd in enumerate(self._optimizer_wd):
                safitty.set(
                    optimizer.param_groups, i, "weight_decay", value=wd)


__all__ = ["OptimizerCallback"]
