from torch.nn.utils import prune

from catalyst.core import Callback, CallbackOrder


class PruningCallback(Callback):
    def __init__(
            self,
            pruner_fn,
            prune_on_epoch_end: bool = False,
            prune_on_stage_end: bool = True,
            remove_reparametrization=True,
            key_to_prune="weight",
            amount=0.5,
    ):
        super().__init__(CallbackOrder.External)
        self.pruner_fn = pruner_fn
        self.prune_on_epoch_end = prune_on_epoch_end
        self.prune_on_stage_end = prune_on_stage_end
        self.remove_reparametrization = remove_reparametrization
        self.key_to_prune = key_to_prune
        self.amount = amount

    def _prune_module(self, module, key):
        self.pruner_fn(module, name=key, amount=self.amount)

    def _run_pruning(self, runner):
        pruned_modules = 0
        for module in runner.model:
            try:
                if isinstance(self.key_to_prune, str):
                    self._prune_module(module, self.key_to_prune)
                elif isinstance(self.key_to_prune, list):
                    for key in self.key_to_prune:
                        self._prune_module(module, key)
                pruned_modules += 1

            except Exception as e:
                pass

        if pruned_modules == 0:
            raise Exception(f"There is no {self.key_to_prune} key in your model")

    def on_epoch_end(self, runner):
        if self.prune_on_epoch_end and runner.num_epochs != runner.epoch:
            self._run_pruning(runner)

    def on_stage_end(self, runner):
        if self.prune_on_stage_end:
            self._run_pruning(runner)
        if self.remove_reparametrization:
            for module in runner.model:
                try:
                    if isinstance(self.key_to_prune, str):
                        prune.remove(module, self.key_to_prune)
                    elif isinstance(self.key_to_prune, list):
                        for key in self.key_to_prune:
                            prune.remove(module, key)
                except Exception as e:
                    pass
