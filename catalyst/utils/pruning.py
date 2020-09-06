from typing import Callable, List, Optional, Union

from torch.nn import Module
from torch.nn.utils import prune

from catalyst.utils.initialization import reset_weights_if_possible


def prune_model(
    model: Module,
    pruning_fn: Callable,
    keys_to_prune: List[str],
    amount: Union[float, int],
    layers_to_prune: Optional[List[str]] = None,
    reinitialize_after_pruning: Optional[bool] = False,
) -> None:
    """
    Prune model function can be used for pruning certain
    tensors in model layers.

    Raises:
        AttributeError: If layers_to_prune is not None, but there is
                no layers with specified name.
        Exception: If no layers have specified keys.

    Args:
        model: Model to be pruned.
        pruning_fn: Pruning function with API same as in
            torch.nn.utils.pruning.
            pruning_fn(module, name, amount).
        keys_to_prune: list of strings. Determines
            which tensor in modules will be pruned.
        amount: quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and
            represent the fraction of parameters to prune.
            If int, it represents the absolute number
            of parameters to prune.
        layers_to_prune: list of strings - module names to be pruned.
            If None provided then will try to prune every module in
            model.
        reinitialize_after_pruning: if True then will reinitialize model
                after pruning. (Lottery Ticket Hypothesis check e.g.)
    """
    pruned_modules = 0
    for name, module in model.named_modules():
        try:
            if layers_to_prune is None or name in layers_to_prune:
                for key in keys_to_prune:
                    pruning_fn(module, name=key, amount=amount)
                pruned_modules += 1
        except AttributeError as e:
            if layers_to_prune is not None:
                raise e

    if pruned_modules == 0:
        raise Exception(f"There is no {keys_to_prune} key in your model")
    if reinitialize_after_pruning:
        model.apply(reset_weights_if_possible)


def remove_reparametrization(
    model: Module,
    keys_to_prune: List[str],
    layers_to_prune: Optional[List[str]] = None,
) -> None:
    """
    Removes pre-hooks and pruning masks from the model.

    Args:
        model: model to remove reparametrization.
        keys_to_prune: list of strings. Determines
            which tensor in modules have already been pruned.
        layers_to_prune: list of strings - module names
            have already been pruned.
            If None provided then will try to prune every module in
            model.
    """
    for name, module in model.named_modules():
        try:
            if layers_to_prune is None or name in layers_to_prune:
                for key in keys_to_prune:
                    prune.remove(module, key)
        except ValueError:
            pass


__all__ = ["prune_model", "remove_reparametrization"]
