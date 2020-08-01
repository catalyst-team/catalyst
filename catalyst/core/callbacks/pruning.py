from typing import Callable, List, Optional, Union
import warnings

import torch
from torch.nn.utils import prune

from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.utils.initialization import weight_reset

PRUNING_FN = {
    "l1_unstructured": prune.l1_unstructured,
    "random_unstructured": prune.random_unstructured,
    "ln_structured": prune.ln_structured,
    "random_structured": prune.random_structured,
}


class PruningCallback(Callback):
    """
    Pruning Callback

    This callback is designed to prune network parameters
    during and/or after training.
    """

    def __init__(
        self,
        pruning_fn: Union[Callable, str],
        keys_to_prune: Optional[List[str]] = None,
        amount: Optional[Union[int, float]] = 0.5,
        prune_on_epoch_end: Optional[bool] = False,
        prune_on_stage_end: Optional[bool] = True,
        remove_reparametrization: Optional[bool] = True,
        reinitialize_after_pruning: Optional[bool] = False,
        layers_to_prune: Optional[List[str]] = None,
        dim: Optional[int] = None,
        n: Optional[int] = None,
    ) -> None:
        """
        Init method for pruning callback

        Args:
            pruning_fn: function from torch.nn.utils.prune module
                or your based on BasePruningMethod. Can be string e.g.
                 `"l1_unstructured"`.
                 See pytorch docs for more details.
            keys_to_prune: list of strings. Determines
                which tensor in modules will be pruned.
            amount: quantity of parameters to prune.
                If float, should be between 0.0 and 1.0 and
                represent the fraction of parameters to prune.
                If int, it represents the absolute number
                of parameters to prune.
            prune_on_epoch_end: bool flag determines call or not
                to call pruning_fn on epoch end.
            prune_on_stage_end: bool flag determines call or not
                to call pruning_fn on stage end.
            remove_reparametrization: if True then all reparametrization
                pre-hooks and tensors with mask will be removed on
                stage end.
            reinitialize_after_pruning: if True then will reinitialize model
                after pruning. (Lottery Ticket Hypothesis)
            layers_to_prune: list of strings - module names to be pruned.
                If None provided then will try to prune every module in
                model.
            dim: if you are using structured pruning method you need
                to specify dimension.
            n: if you are using ln_structured you need to specify l_n
                norm.

        """
        super().__init__(CallbackOrder.External)
        if isinstance(pruning_fn, str):
            if pruning_fn not in PRUNING_FN.keys():
                raise Exception(
                    f"Pruning function should be in {PRUNING_FN.keys()}, "
                    "global pruning is not currently support."
                )
            if "unstructured" not in pruning_fn:
                if dim is None:
                    raise Exception(
                        "If you are using structured pruning you"
                        "need to specify dim in callback args"
                    )
                if pruning_fn == "ln_structured":
                    if n is None:
                        raise Exception(
                            "If you are using ln_unstructured you"
                            "need to specify n in callback args"
                        )
                    self.pruning_fn = self._wrap_pruning_fn(
                        prune.ln_structured, dim=dim, n=n
                    )
                else:
                    self.pruning_fn = self._wrap_pruning_fn(
                        PRUNING_FN[pruning_fn], dim=dim
                    )
            else:  # unstructured
                self.pruning_fn = PRUNING_FN[pruning_fn]
        else:
            self.pruning_fn = pruning_fn
        if keys_to_prune is None:
            keys_to_prune = ["weight"]
        self.prune_on_epoch_end = prune_on_epoch_end
        self.prune_on_stage_end = prune_on_stage_end
        if not (prune_on_stage_end or prune_on_epoch_end):
            warnings.warn(
                "Warning!"
                "You disabled pruning pruning both on epoch and stage end."
                "Model won't be pruned by this callback."
            )
        self.remove_reparametrization = remove_reparametrization
        self.keys_to_prune = keys_to_prune
        self.amount = amount
        self.reinitialize_after_pruning = reinitialize_after_pruning
        self.layers_to_prune = layers_to_prune

    @staticmethod
    def _wrap_pruning_fn(pruning_fn, *args, **kwargs):
        return lambda module, name, amount: pruning_fn(
            module, name, amount, *args, **kwargs
        )

    def _prune_module(self, module):
        for key in self.keys_to_prune:
            self.pruning_fn(module, name=key, amount=self.amount)

    def _to_be_pruned(self, layer_name):
        return (
            self.layers_to_prune is None or layer_name in self.layers_to_prune
        )

    def _run_pruning(self, model: torch.nn.Module):
        pruned_modules = 0
        for name, module in model.named_modules():
            try:
                if self._to_be_pruned(name):
                    self._prune_module(module)
                    pruned_modules += 1
            except AttributeError as e:
                if self.layers_to_prune is not None:
                    raise e

        if pruned_modules == 0:
            raise Exception(
                f"There is no {self.keys_to_prune} key in your model"
            )
        if self.reinitialize_after_pruning:
            model.apply(weight_reset)

    def _remove_reparametrization(self, runner: "IRunner"):
        for name, module in runner.model.named_modules():
            try:
                if self._to_be_pruned(name):
                    if isinstance(self.keys_to_prune, str):
                        prune.remove(module, self.keys_to_prune)
                    elif isinstance(self.keys_to_prune, list):
                        for key in self.keys_to_prune:
                            prune.remove(module, key)
            except ValueError:
                pass

    def on_epoch_end(self, runner: "IRunner") -> None:
        """
        On epoch end action.

        Active if prune_on_epoch_end is True.
        Args:
            runner: runner for your experiment
        """
        if self.prune_on_epoch_end and runner.num_epochs != runner.epoch:
            self._run_pruning(runner.model)

    def on_stage_end(self, runner: "IRunner") -> None:
        """
        On stage end action.

        Active if prune_on_stage_end or
        remove_reparametrization is True.
        Args:
            runner: runner for your experiment
        """
        if self.prune_on_stage_end:
            self._run_pruning(runner.model)
        if self.remove_reparametrization:
            self._remove_reparametrization(runner)
