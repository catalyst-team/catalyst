from typing import Callable, List, Optional, Union

from torch.nn import Module
from torch.nn.utils import prune

from catalyst.utils.torch import get_nn_from_ddp_module

PRUNING_FN = {  # noqa: WPS407
    "l1_unstructured": prune.l1_unstructured,
    "random_unstructured": prune.random_unstructured,
    "ln_structured": prune.ln_structured,
    "random_structured": prune.random_structured,
}


def _wrap_pruning_fn(pruning_fn, *args, **kwargs):
    return lambda module, name, amount: pruning_fn(module, name, amount, *args, **kwargs)


def get_pruning_fn(
    pruning_fn: Union[str, Callable], dim: int = None, l_norm: int = None
) -> Callable:
    """[summary]

    Args:
        pruning_fn (Union[str, Callable]): function from torch.nn.utils.prune module
                or your based on BasePruningMethod. Can be string e.g.
                `"l1_unstructured"`. See pytorch docs for more details.
        dim (int, optional): if you are using structured pruning method you need
                to specify dimension. Defaults to None.
        l_norm (int, optional): if you are using ln_structured you need to specify l_norm.
            Defaults to None.

    Raises:
        ValueError: If ``dim`` or ``l_norm`` is not defined when it's required.

    Returns:
        Callable: pruning_fn
    """
    if isinstance(pruning_fn, str):
        if pruning_fn not in PRUNING_FN.keys():
            raise ValueError(
                f"Pruning function should be in {PRUNING_FN.keys()}, "
                "global pruning is not currently support."
            )
        if "unstructured" not in pruning_fn:
            if dim is None:
                raise ValueError(
                    "If you are using structured pruning you" "need to specify dim in args"
                )
            if pruning_fn == "ln_structured":
                if l_norm is None:
                    raise ValueError(
                        "If you are using ln_unstructured you" "need to specify l_norm in args"
                    )
                pruning_fn = _wrap_pruning_fn(prune.ln_structured, dim=dim, n=l_norm)
            else:
                pruning_fn = _wrap_pruning_fn(PRUNING_FN[pruning_fn], dim=dim)
        else:  # unstructured
            pruning_fn = PRUNING_FN[pruning_fn]
    return pruning_fn


def prune_model(
    model: Module,
    pruning_fn: Union[Callable, str],
    amount: Union[float, int],
    keys_to_prune: Optional[List[str]] = None,
    layers_to_prune: Optional[List[str]] = None,
    dim: int = None,
    l_norm: int = None,
) -> None:
    """
    Prune model function can be used for pruning certain
    tensors in model layers.

    Args:
        model: Model to be pruned.
        pruning_fn: Pruning function with API same as in torch.nn.utils.pruning.
            pruning_fn(module, name, amount).
        keys_to_prune: list of strings. Determines which tensor in modules will be pruned.
        amount: quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and
            represent the fraction of parameters to prune.
            If int, it represents the absolute number
            of parameters to prune.
        layers_to_prune: list of strings - module names to be pruned.
            If None provided then will try to prune every module in model.
        dim (int, optional): if you are using structured pruning method you need
            to specify dimension. Defaults to None.
        l_norm (int, optional): if you are using ln_structured you need to specify l_norm.
            Defaults to None.

    Example:
        .. code-block:: python

           pruned_model = prune_model(model, pruning_fn="l1_unstructured")

    Raises:
        AttributeError: If layers_to_prune is not None, but there is
            no layers with specified name. OR
        ValueError: if no layers have specified keys.
    """
    nn_model = get_nn_from_ddp_module(model)
    pruning_fn = get_pruning_fn(pruning_fn, l_norm=l_norm, dim=dim)
    keys_to_prune = keys_to_prune or ["weight"]
    pruned_modules = 0
    for name, module in nn_model.named_modules():
        try:
            if layers_to_prune is None or name in layers_to_prune:
                for key in keys_to_prune:
                    pruning_fn(module, name=key, amount=amount)
                pruned_modules += 1
        except AttributeError as e:
            if layers_to_prune is not None:
                raise e

    if pruned_modules == 0:
        raise ValueError(f"There is no {keys_to_prune} key in your model")


def remove_reparametrization(
    model: Module, keys_to_prune: List[str], layers_to_prune: Optional[List[str]] = None,
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
    nn_model = get_nn_from_ddp_module(model)
    for name, module in nn_model.named_modules():
        try:
            if layers_to_prune is None or name in layers_to_prune:
                for key in keys_to_prune:
                    prune.remove(module, key)
        except ValueError:
            pass


__all__ = ["prune_model", "remove_reparametrization", "get_pruning_fn"]
