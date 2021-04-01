from typing import Callable, List, Union

from catalyst.core import Callback, CallbackOrder


class LambdaPreprocessCallback(Callback):
    """
    Preprocess your batch with specified function.

    Args:
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
                lambda_fn=lambda x: x.argmax(-1)
            )
            runner.train(
                model=model,
                callbacks=[accuracy_callback],
                loaders=loaders,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=1,
                logdir="./logs",
            )

    """

    def __init__(
        self,
        lambda_fn: Callable,
        keys_to_apply: Union[List[str], str] = "logits",
    ):
        """Wraps input for your callback with specified function.

        Args:
            lambda_fn (Callable): Function to apply.
            keys_to_apply (Union[List[str], str], optional): Keys in batch dict to apply function.
                Defaults to ["s_hidden_states", "t_hidden_states"].

        Raises:
            TypeError: When keys_to_apply is not str or list.
        """
        super().__init__(order=CallbackOrder.Internal)
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


__all__ = ["LambdaPreprocessCallback"]
