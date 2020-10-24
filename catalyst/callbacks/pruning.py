from typing import Callable, List, Optional, TYPE_CHECKING, Union
import warnings

from torch.nn.utils import prune

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.utils.pruning import prune_model, remove_reparametrization

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner

PRUNING_FN = {  # noqa: WPS407
    "l1_unstructured": prune.l1_unstructured,
    "random_unstructured": prune.random_unstructured,
    "ln_structured": prune.ln_structured,
    "random_structured": prune.random_structured,
}


def _wrap_pruning_fn(pruning_fn, *args, **kwargs):
    return lambda module, name, amount: pruning_fn(
        module, name, amount, *args, **kwargs
    )


class PruningCallback(Callback):
    """
    Pruning Callback

    This callback is designed to prune network parameters
    during and/or after training.

    Args:
        pruning_fn: function from torch.nn.utils.prune module
            or your based on BasePruningMethod. Can be string e.g.
            `"l1_unstructured"`. See pytorch docs for more details.
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
        remove_reparametrization_on_stage_end: if True then all
            reparametrization pre-hooks and tensors with mask
            will be removed on stage end.
        reinitialize_after_pruning: if True then will reinitialize model
            after pruning. (Lottery Ticket Hypothesis)
        layers_to_prune: list of strings - module names to be pruned.
            If None provided then will try to prune every module in
            model.
        dim: if you are using structured pruning method you need
            to specify dimension.
        l_norm: if you are using ln_structured you need to specify l_norm.
    """

    def __init__(
        self,
        pruning_fn: Union[Callable, str],
        keys_to_prune: Optional[List[str]] = None,
        amount: Optional[Union[int, float]] = 0.5,
        prune_on_epoch_end: Optional[bool] = False,
        prune_on_stage_end: Optional[bool] = True,
        remove_reparametrization_on_stage_end: Optional[bool] = True,
        reinitialize_after_pruning: Optional[bool] = False,
        layers_to_prune: Optional[List[str]] = None,
        dim: Optional[int] = None,
        l_norm: Optional[int] = None,
    ) -> None:
        """Init method for pruning callback"""
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
                    if l_norm is None:
                        raise Exception(
                            "If you are using ln_unstructured you"
                            "need to specify n in callback args"
                        )
                    self.pruning_fn = _wrap_pruning_fn(
                        prune.ln_structured, dim=dim, n=l_norm
                    )
                else:
                    self.pruning_fn = _wrap_pruning_fn(
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
        self.remove_reparametrization_on_stage_end = (
            remove_reparametrization_on_stage_end
        )
        self.keys_to_prune = keys_to_prune
        self.amount = amount
        self.reinitialize_after_pruning = reinitialize_after_pruning
        self.layers_to_prune = layers_to_prune

    def on_epoch_end(self, runner: "IRunner") -> None:
        """
        On epoch end action.

        Active if prune_on_epoch_end is True.

        Args:
            runner: runner for your experiment
        """
        if self.prune_on_epoch_end and runner.num_epochs != runner.epoch:
            prune_model(
                model=runner.model,
                pruning_fn=self.pruning_fn,
                keys_to_prune=self.keys_to_prune,
                amount=self.amount,
                layers_to_prune=self.layers_to_prune,
                reinitialize_after_pruning=self.reinitialize_after_pruning,
            )

    def on_stage_end(self, runner: "IRunner") -> None:
        """
        On stage end action.

        Active if prune_on_stage_end or
        remove_reparametrization is True.

        Args:
            runner: runner for your experiment
        """
        if self.prune_on_stage_end:
            prune_model(
                model=runner.model,
                pruning_fn=self.pruning_fn,
                keys_to_prune=self.keys_to_prune,
                amount=self.amount,
                layers_to_prune=self.layers_to_prune,
                reinitialize_after_pruning=self.reinitialize_after_pruning,
            )
        if self.remove_reparametrization_on_stage_end:
            remove_reparametrization(
                model=runner.model,
                keys_to_prune=self.keys_to_prune,
                layers_to_prune=self.layers_to_prune,
            )


__all__ = ["PruningCallback"]
