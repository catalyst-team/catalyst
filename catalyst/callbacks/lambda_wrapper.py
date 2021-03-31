from typing import Callable, List, Union
from copy import deepcopy

from catalyst.core import Callback


class LambdaWrapperCallback(Callback):
    """
    Wraps input for your callback with specified function.

    Args:
        base_callback (Callback): Base callback.
        lambda_fn (Callable): Function to apply.
        keys_to_apply (Union[List[str], str], optional): Keys in batch dict to apply function.
            Defaults to ["s_hidden_states", "t_hidden_states"].

    Raises:
        TypeError: When keys_to_apply is not str or list.

    Examples:
        .. code-block:: python
            import os

            import torch
            from torch import nn
            from torch.utils.data import DataLoader

            from catalyst import dl
            from catalyst.data.transforms import ToTensor
            from catalyst.contrib.datasets import MNIST
            from catalyst.contrib.nn.modules import Flatten

            loaders = {
                "train": DataLoader(
                    MNIST(
                        os.getcwd(), train=False, download=True, transform=ToTensor()
                    ),
                    batch_size=32,
                ),
                "valid": DataLoader(
                    MNIST(
                        os.getcwd(), train=False, download=True, transform=ToTensor()
                    ),
                    batch_size=32,
                ),
            }

            model = nn.Sequential(
                Flatten(),
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            runner = dl.SupervisedRunner()
            accuracy_callback = dl.LambdaWrapperCallback(
                base_callback=dl.AccuracyCallback(input_key="logits", target_key="targets"),
                lambda_fn=lambda x: x.argmax(-1)
            )
            runner.train(
                model=model,
                callbacks=[dl.AccuracyCallback(input_key="logits", )],
                loaders=loaders,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=1,
                logdir="./logs",
            )
    """

    def __init__(
        self,
        base_callback: Callback,
        lambda_fn: Callable,
        keys_to_apply: Union[List[str], str] = "logits",
    ):
        """Wraps input for your callback with specified function.

        Args:
            base_callback (Callback): Base callback.
            lambda_fn (Callable): Function to apply.
            keys_to_apply (Union[List[str], str], optional): Keys in batch dict to apply function.
                Defaults to ["s_hidden_states", "t_hidden_states"].

        Raises:
            TypeError: When keys_to_apply is not str or list.
        """
        super().__init__(order=base_callback.order)
        self.base_callback = base_callback
        if not isinstance(keys_to_apply, (list, str)):
            raise TypeError("keys to apply should be str or list of str.")
        self.keys_to_apply = keys_to_apply
        self.lambda_fn = lambda_fn

    def on_batch_end(self, runner) -> None:
        """
        On batch end action.

        Args:
            runner: runner for the experiment.

        Raises:
            TypeError: If lambda_fn output has a wrong type.

        """
        orig_batch = deepcopy(runner.batch)
        batch = runner.batch

        if isinstance(self.keys_to_apply, list):
            fn_inp = [batch[key] for key in self.keys_to_apply]
            fn_output = self.lambda_fn(*fn_inp)
            if isinstance(fn_output, tuple):
                for idx, key in enumerate(self.keys_to_apply):
                    batch[key] = fn_output[idx]
            elif isinstance(fn_output, dict):
                for outp_k, outp_v in fn_output.items():
                    batch[outp_k] = outp_v
            else:
                raise TypeError(
                    "If keys_to_apply is list, then function output should be tuple or dict."
                )
        elif isinstance(self.keys_to_apply, str):
            batch[self.keys_to_apply] = self.lambda_fn(self.keys_to_apply)
        runner.batch = batch
        self.base_callback.on_batch_end(runner)
        runner.batch = orig_batch


__all__ = ["LambdaWrapperCallback"]
