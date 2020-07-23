from typing import Callable, List, Optional, Union
import warnings

from torch.nn.utils import prune

from catalyst.core import Callback, CallbackOrder, IRunner


class PruningCallback(Callback):
    """
    Pruning Callback

    This callback is designed to prune network parameters
    during and/or after training.
    """

    def __init__(
        self,
        pruner_fn: Callable,
        key_to_prune: Union[str, List[str]] = "weight",
        amount: Union[int, float] = 0.5,
        prune_on_epoch_end: bool = False,
        prune_on_stage_end: bool = True,
        remove_reparametrization: bool = True,
        reinitialize_after_pruning: bool = False,
        layers_to_prune: Optional[List[str]] = None,
    ) -> None:
        """
        Init method for pruning callback

        Args:
            pruner_fn: function from torch.nn.utils.prune module
                or your based on BasePruningMethod. See pytorch
                docs for more details.
            key_to_prune: can be string or list of strings. Determines
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

        """
        super().__init__(CallbackOrder.External)
        self.pruner_fn = pruner_fn
        self.prune_on_epoch_end = prune_on_epoch_end
        self.prune_on_stage_end = prune_on_stage_end
        if not (prune_on_stage_end or prune_on_epoch_end):
            warnings.warn(
                "Warning!"
                "You disabled pruning pruning both on epoch and stage end."
                "Model won't be pruned by this callback."
            )
        self.remove_reparametrization = remove_reparametrization
        self.key_to_prune = key_to_prune
        self.amount = amount
        self.reinitialize_after_pruning = reinitialize_after_pruning
        self.layers_to_prune = layers_to_prune

    @staticmethod
    def _weight_reset(m):
        try:
            m.reset_parameters()
        except AttributeError:
            pass

    def _prune_module(self, module):
        if isinstance(self.key_to_prune, str):
            self.pruner_fn(module, name=self.key_to_prune, amount=self.amount)
        elif isinstance(self.key_to_prune, list):
            for key in self.key_to_prune:
                self.pruner_fn(module, name=key, amount=self.amount)

    def _to_be_pruned(self, layer_name):
        if self.layers_to_prune is None:
            return True
        return layer_name in self.layers_to_prune

    def _run_pruning(self, runner: "IRunner"):
        pruned_modules = 0
        for name, module in runner.model.named_modules():
            try:
                if self._to_be_pruned(name):
                    self._prune_module(module)
                    pruned_modules += 1
            except AttributeError as e:
                if self.layers_to_prune is not None:
                    raise e

        if pruned_modules == 0:
            raise Exception(
                f"There is no {self.key_to_prune} key in your model"
            )
        if self.reinitialize_after_pruning:
            runner.model.apply(self._weight_reset)

    def _remove_reparametrization(self, runner: "IRunner"):
        for name, module in runner.model.named_modules():
            try:
                if self._to_be_pruned(name):
                    if isinstance(self.key_to_prune, str):
                        prune.remove(module, self.key_to_prune)
                    elif isinstance(self.key_to_prune, list):
                        for key in self.key_to_prune:
                            prune.remove(module, key)
            except ValueError:
                pass

    def on_epoch_end(self, runner) -> None:
        """
        On epoch end action.

        Active if prune_on_epoch_end is True.
        Args:
            runner: runner for your experiment
        """
        if self.prune_on_epoch_end and runner.num_epochs != runner.epoch:
            self._run_pruning(runner)

    def on_stage_end(self, runner) -> None:
        """
        On stage end action.

        Active if prune_on_stage_end or
        remove_reparametrization is True.
        Args:
            runner: runner for your experiment
        """
        if self.prune_on_stage_end:
            self._run_pruning(runner)
        if self.remove_reparametrization:
            self._remove_reparametrization(runner)
