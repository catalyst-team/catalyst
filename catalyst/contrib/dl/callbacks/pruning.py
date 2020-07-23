from typing import Callable, List, Union

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

        """
        super().__init__(CallbackOrder.External)
        self.pruner_fn = pruner_fn
        self.prune_on_epoch_end = prune_on_epoch_end
        self.prune_on_stage_end = prune_on_stage_end
        self.remove_reparametrization = remove_reparametrization
        self.key_to_prune = key_to_prune
        self.amount = amount
        self.reinitialize_after_pruning = reinitialize_after_pruning

    @staticmethod
    def _weight_reset(m):
        try:
            m.reset_parameters()
        except AttributeError:
            pass

    def _prune_module(self, module, key):
        self.pruner_fn(module, name=key, amount=self.amount)

    def _run_pruning(self, runner: "IRunner"):
        pruned_modules = 0
        for module in runner.model:
            try:
                if isinstance(self.key_to_prune, str):
                    self._prune_module(module, self.key_to_prune)
                elif isinstance(self.key_to_prune, list):
                    for key in self.key_to_prune:
                        self._prune_module(module, key)
                pruned_modules += 1

            except AttributeError:
                pass

        if pruned_modules == 0:
            raise Exception(
                f"There is no {self.key_to_prune} key in your model"
            )
        if self.reinitialize_after_pruning:
            runner.model.apply(self._weight_reset)

    def _remove_reparametrization(self, runner: "IRunner"):
        for module in runner.model:
            try:
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
