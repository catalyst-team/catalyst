from typing import Callable, List, Optional, TYPE_CHECKING, Union
import warnings

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.utils.pruning import get_pruning_fn, prune_model, remove_reparametrization

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class PruningCallback(Callback):
    """This callback prunes network parameters during and/or after training.

    Args:
        pruning_fn: function from torch.nn.utils.prune module
            or your based on BasePruningMethod. Can be string e.g.
            `"l1_unstructured"`. See pytorch docs for more details.
        amount: quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and
            represent the fraction of parameters to prune.
            If int, it represents the absolute number
            of parameters to prune.
        keys_to_prune: list of strings. Determines
            which tensor in modules will be pruned.
        prune_on_epoch_end: bool flag determines call or not
            to call pruning_fn on epoch end.
        prune_on_stage_end: bool flag determines call or not
            to call pruning_fn on stage end.
        remove_reparametrization_on_stage_end: if True then all
            reparametrization pre-hooks and tensors with mask
            will be removed on stage end.
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
        amount: Union[int, float],
        keys_to_prune: Optional[List[str]] = None,
        prune_on_epoch_end: Optional[bool] = False,
        prune_on_stage_end: Optional[bool] = True,
        remove_reparametrization_on_stage_end: Optional[bool] = True,
        layers_to_prune: Optional[List[str]] = None,
        dim: Optional[int] = None,
        l_norm: Optional[int] = None,
    ) -> None:
        """Init."""
        super().__init__(CallbackOrder.ExternalExtra)
        self.pruning_fn = get_pruning_fn(pruning_fn=pruning_fn, dim=dim, l_norm=l_norm)
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
        self.remove_reparametrization_on_stage_end = remove_reparametrization_on_stage_end
        self.keys_to_prune = keys_to_prune
        self.amount = amount
        self.layers_to_prune = layers_to_prune

    def on_epoch_end(self, runner: "IRunner") -> None:
        """Event handler.

        Active if prune_on_epoch_end is True.

        Args:
            runner: runner for your experiment
        """
        if self.prune_on_epoch_end and runner.stage_epoch_step != runner.stage_epoch_len:
            prune_model(
                model=runner.model,
                pruning_fn=self.pruning_fn,
                keys_to_prune=self.keys_to_prune,
                amount=self.amount,
                layers_to_prune=self.layers_to_prune,
            )

    def on_stage_end(self, runner: "IRunner") -> None:
        """Event handler.

        Active if prune_on_stage_end or remove_reparametrization is True.

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
            )
        if self.remove_reparametrization_on_stage_end:
            remove_reparametrization(
                model=runner.model,
                keys_to_prune=self.keys_to_prune,
                layers_to_prune=self.layers_to_prune,
            )


__all__ = ["PruningCallback"]
