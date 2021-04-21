from typing import Any, List, Mapping, Tuple, Union

import torch

from catalyst.core.runner import IRunner


class ISupervisedRunner(IRunner):
    """IRunner for experiments with supervised model.

    Args:
        input_key: key in ``runner.batch`` dict mapping for model input
        output_key: key for ``runner.batch`` to store model output
        target_key: key in ``runner.batch`` dict mapping for target
        loss_key: key for ``runner.batch_metrics`` to store criterion loss output

    Abstraction, please check out implementations for more details:

        - :py:mod:`catalyst.runners.runner.SupervisedRunner`
        - :py:mod:`catalyst.runners.config.SupervisedConfigRunner`
        - :py:mod:`catalyst.runners.hydra.SupervisedHydraRunner`

    .. note::
        ISupervisedRunner contains only the logic with batch handling.


    ISupervisedRunner logic pseudocode:

    .. code-block:: python

        batch = {"input_key": tensor, "target_key": tensor}
        output = model(batch["input_key"])
        batch["output_key"] = output
        loss = criterion(batch["output_key"], batch["target_key"])
        batch_metrics["loss_key"] = loss



    .. note::
        Please follow the `minimal examples`_ sections for use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples

    Examples:

    .. code-block:: python

        import os
        from torch import nn, optim
        from torch.utils.data import DataLoader
        from catalyst import dl, utils
        from catalyst.data.transforms import ToTensor
        from catalyst.contrib.datasets import MNIST

        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()),
                batch_size=32
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                batch_size=32
            ),
        }

        runner = dl.SupervisedRunner(
            input_key="features", output_key="logits", target_key="targets", loss_key="loss"
        )
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            callbacks=[
                dl.AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1, 3)),
                dl.PrecisionRecallF1SupportCallback(
                    input_key="logits", target_key="targets", num_classes=10
                ),
                dl.AUCCallback(input_key="logits", target_key="targets"),
            ],
            logdir="./logs",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
            load_best_on_end=True,
        )
        # model inference
        for prediction in runner.predict_loader(loader=loaders["valid"]):
            assert prediction["logits"].detach().cpu().numpy().shape[-1] == 10
    """

    def __init__(
        self,
        input_key: Any = "features",
        output_key: Any = "logits",
        target_key: str = "targets",
        loss_key: str = "loss",
    ):
        """Init."""
        IRunner.__init__(self)

        self._input_key = input_key
        self._output_key = output_key
        self._target_key = target_key
        self._loss_key = loss_key

        if isinstance(self._input_key, str):
            # when model expects value
            self._process_input = self._process_input_str
        elif isinstance(self._input_key, (list, tuple)):
            # when model expects tuple
            self._process_input = self._process_input_list
        elif self._input_key is None:
            # when model expects dict
            self._process_input = self._process_input_none
        else:
            raise NotImplementedError()

        if isinstance(output_key, str):
            # when model returns value
            self._process_output = self._process_output_str
        elif isinstance(output_key, (list, tuple)):
            # when model returns tuple
            self._process_output = self._process_output_list
        elif self._output_key is None:
            # when model returns dict
            self._process_output = self._process_output_none
        else:
            raise NotImplementedError()

    def _process_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 2
            batch = {self._input_key: batch[0], self._target_key: batch[1]}
        return batch

    def _process_input_str(self, batch: Mapping[str, Any], **kwargs):
        output = self.model(batch[self._input_key], **kwargs)
        return output

    def _process_input_list(self, batch: Mapping[str, Any], **kwargs):
        input = {key: batch[key] for key in self._input_key}  # noqa: WPS125
        output = self.model(**input, **kwargs)
        return output

    def _process_input_none(self, batch: Mapping[str, Any], **kwargs):
        output = self.model(**batch, **kwargs)
        return output

    def _process_output_str(self, output: torch.Tensor):
        output = {self._output_key: output}
        return output

    def _process_output_list(self, output: Union[Tuple, List]):
        output = {key: value for key, value in zip(self._output_key, output)}
        return output

    def _process_output_none(self, output: Mapping[str, Any]):
        return output

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Forward method for your Runner.
        Should not be called directly outside of runner.
        If your model has specific interface, override this method to use it

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoaders.
            **kwargs: additional parameters to pass to the model

        Returns:
            dict with model output batch
        """
        output = self._process_input(batch, **kwargs)
        output = self._process_output(output)
        return output

    def on_batch_start(self, runner: "IRunner"):
        """Event handler."""
        self.batch = self._process_batch(self.batch)
        super().on_batch_start(runner)

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        """
        Inner method to handle specified data batch.
        Used to make a train/valid/infer stage during Experiment run.

        Args:
            batch: dictionary with data batches from DataLoader.
        """
        self.batch = {**batch, **self.forward(batch)}


__all__ = ["ISupervisedRunner"]
