from typing import Any, Mapping, Dict, List, Union
from collections import OrderedDict  # noqa F401

from torch import nn
from torch.utils.data import DataLoader  # noqa F401

from catalyst.dl.core import Runner, Callback
from catalyst.dl.experiment import SupervisedExperiment
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
from catalyst.dl.utils.torch import _Model, _Criterion, _Optimizer, _Scheduler


class SupervisedRunner(Runner):
    """
    Runner for experiments with supervised model
    """
    _default_experiment = SupervisedExperiment

    def __init__(
        self,
        model: nn.Module = None,
        device=None,
        input_key: str = "features",
        output_key: str = "logits",
        input_target_key: str = "targets",
    ):
        """
        @TODO update docs
        Args:
            input_key: Key in batch dict mapping to model input
            output_key: Key in output dict model output will be stored under
        """
        super().__init__(model=model, device=device)
        self.input_key = input_key
        self.output_key = output_key
        self.target_key = input_target_key

        if isinstance(self.input_key, str):
            self._process_input = self._process_input_str
        elif isinstance(self.input_key, (list, tuple)):
            self._process_input = self._process_input_list
        else:
            self._process_input = self._process_input_none

        if isinstance(output_key, str):
            self._process_output = self._process_output_str
        elif isinstance(output_key, (list, tuple)):
            self._process_output = self._process_output_list
        else:
            self._process_output = self._process_output_none

    def _batch2device(self, batch: Mapping[str, Any], device):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 2
            batch = {self.input_key: batch[0], self.target_key: batch[1]}
        batch = super()._batch2device(batch, device)
        return batch

    def _process_input_str(self, batch: Mapping[str, Any]):
        output = self.model(batch[self.input_key])
        return output

    def _process_input_list(self, batch: Mapping[str, Any]):
        input = dict((key, batch[key]) for key in self.input_key)
        output = self.model(**input)
        return output

    def _process_input_none(self, batch: Mapping[str, Any]):
        output = self.model(**batch)
        return output

    def _process_output_str(self, output: Mapping[str, Any]):
        output = {self.output_key: output}
        return output

    def _process_output_list(self, output: Mapping[str, Any]):
        output = dict(
            (key, value) for key, value in zip(self.output_key, output)
        )
        return output

    def _process_output_none(self, output: Mapping[str, Any]):
        return output

    def predict_batch(self, batch: Mapping[str, Any]):
        output = self._process_input(batch)
        output = self._process_output(output)
        return output

    def train(
        self,
        model: _Model,
        criterion: _Criterion,
        optimizer: _Optimizer,
        loaders: "OrderedDict[str, DataLoader]",
        logdir: str,
        callbacks: "Union[List[Callback], OrderedDict[str, Callback]]" = None,
        scheduler: _Scheduler = None,
        num_epochs: int = 1,
        valid_loader: str = "valid",
        main_metric: str = "loss",
        minimize_metric: bool = True,
        verbose: bool = False,
        state_kwargs: Dict = None,
        checkpoint_data: Dict = None,
        fp16: Union[Dict, bool] = None,
        check: bool = False,
    ):
        if isinstance(fp16, bool) and fp16:
            fp16 = {"opt_level": "O1"}
        experiment = self._default_experiment(
            stage="train",
            model=model,
            loaders=loaders,
            callbacks=callbacks,
            logdir=logdir,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            valid_loader=valid_loader,
            main_metric=main_metric,
            minimize_metric=minimize_metric,
            verbose=verbose,
            state_kwargs=state_kwargs,
            checkpoint_data=checkpoint_data,
            distributed_params=fp16
        )
        self.run_experiment(experiment, check=check)

    def infer(
        self,
        model: _Model,
        loaders: "OrderedDict[str, DataLoader]",
        callbacks: "Union[List[Callback], OrderedDict[str, Callback]]" = None,
        verbose: bool = False,
        state_kwargs: Dict = None,
        fp16: Union[Dict, bool] = None,
        check: bool = False,
    ):
        if isinstance(fp16, bool) and fp16:
            fp16 = {"opt_level": "O1"}
        experiment = self._default_experiment(
            stage="infer",
            model=model,
            loaders=loaders,
            callbacks=callbacks,
            verbose=verbose,
            state_kwargs=state_kwargs,
            distributed_params=fp16
        )
        self.run_experiment(experiment, check=check)

    def predict_loader(
        self,
        loader: DataLoader,
        resume: str = None,
        verbose: bool = False,
        state_kwargs: Dict = None,
        fp16: Union[Dict, bool] = None,
        check: bool = False,
    ):
        loaders = OrderedDict([("infer", loader)])

        callbacks = OrderedDict([("inference", InferCallback())])
        if resume is not None:
            callbacks["loader"] = CheckpointCallback(resume=resume)

        self.infer(
            model=self.model,
            loaders=loaders,
            callbacks=callbacks,
            verbose=verbose,
            state_kwargs=state_kwargs,
            fp16=fp16,
            check=check
        )

        output = callbacks["inference"].predictions
        if isinstance(self.output_key, str):
            output = output[self.output_key]

        return output


__all__ = ["SupervisedRunner"]
