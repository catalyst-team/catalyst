from typing import Callable, List, Optional, Union

from torch.nn import Module
from torch.nn.utils import prune

from catalyst.utils.initialization import weight_reset


def prune_model(
    model: Module,
    pruning_fn: Callable,
    keys_to_prune: List[str],
    amount: Union[float, int],
    layers_to_prune: Optional[List[str]] = None,
    reinitialize_after_pruning: Optional[bool] = False,
) -> None:
    """
    @TODO
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
        model.apply(weight_reset)


def remove_reparametrization(
    model: Module,
    keys_to_prune: List[str],
    layers_to_prune: Optional[List[str]] = None,
) -> None:
    """
    @TODO
    """
    for name, module in model.named_modules():
        try:
            if layers_to_prune is None or name in layers_to_prune:
                for key in keys_to_prune:
                    prune.remove(module, key)
        except ValueError:
            pass
