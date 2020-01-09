from typing import Iterable, Dict, Mapping, Any, Union, OrderedDict
from torch.utils.data import DataLoader
from catalyst.dl.core import Experiment
from catalyst.dl.callbacks import (
    CheckpointCallback,
    ConsoleLogger,
    RaiseExceptionCallback,
    VerboseLogger,
)
from catalyst.dl.utils import process_callbacks
from catalyst.utils.typing import (
    Model,
    Criterion,
    Optimizer,
    Scheduler,
    Dataset
)


class GanExperiment(Experiment):
    def __init__(
        self,
        models: Union[Model, Dict[str, Model]],
        criterions: Dict[str, Criterion],
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Scheduler] = {},
        callbacks: "Dict[str, Callback]" = {},  # noqa: F821, E501
        datasets: Dict[str, Any] = {},
        loaders: Dict[str, Any] = {},
        state_params: Dict[str, Dict[str, Any]] = {},
        monitoring_params: Dict = {},
        distributed_params: Dict = {},
        stages: Iterable[str] = ["train"],
        logdir: str = "./logs",
        verbose: bool = False,
        initial_seed: int = 42,
    ):
        self._models = models
        self._criterions = criterions
        self._optimizers = optimizers
        self._schedulers = schedulers
        # Process callbacks for every stage
        for stage, stage_callbacks in callbacks.items():
            callbacks[stage] = process_callbacks(stage_callbacks)
        self._callbacks = callbacks
        self._datasets = datasets
        self._loaders = loaders
        self._state_params = state_params
        self._monitoring_params = monitoring_params
        self._distributed_params = distributed_params
        self._stages = stages
        self._logdir = logdir
        self._verbose = verbose
        self._initial_seed = initial_seed

    @property
    def initial_seed(self) -> int:
        return self._initial_seed

    @property
    def logdir(self) -> str:
        return self._logdir

    @property
    def stages(self) -> Iterable[str]:
        return self._stages

    @property
    def distributed_params(self) -> Dict:
        return self._distributed_params

    @property
    def monitoring_params(self) -> Dict:
        return self._monitoring_params

    def get_state_params(self, stage: str) -> Mapping[str, Any]:
        return self._state_params[stage]

    def get_model(self, stage: str) -> Model:
        return self._models[stage]

    def get_criterion(self, stage: str) -> Criterion:
        return self._criterions[stage]

    def get_optimizer(self, stage: str, model: Model) -> Optimizer:
        return self._optimizers[stage]

    def get_scheduler(self, stage: str, optimizer: Optimizer) -> Scheduler:
        return self._schedulers.get(stage, None)

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":  # noqa: F821, E501
        callbacks = self._callbacks[stage]
        default_callbacks = []
        if self._verbose:
            default_callbacks.append(("verbose", VerboseLogger))
        if not stage.startswith("infer"):
            default_callbacks.append(("saver", CheckpointCallback))
            default_callbacks.append(("console", ConsoleLogger))
        default_callbacks.append(("exception", RaiseExceptionCallback))
        # Check for absent callbacks and add them
        for callback_name, callback_fn in default_callbacks:
            is_already_present = any(
                isinstance(x, callback_fn) for x in callbacks.values()
            )
            if not is_already_present:
                callbacks[callback_name] = callback_fn()
        return callbacks

    def get_datasets(self, stage: str, **kwargs) -> OrderedDict[str, Dataset]:
        return self._datasets.get(stage, None)

    def get_loaders(self, stage: str) -> OrderedDict[str, DataLoader]:
        return self._loaders.get(stage, None)

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        return NotImplementedError(
            "No static transforms are used in GAN experiment"
        )


__all__ = ["GanExperiment"]
