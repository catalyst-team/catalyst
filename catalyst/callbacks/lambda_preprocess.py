from typing import Any, Callable, Dict, List, Tuple, Union

from catalyst.core import Callback, CallbackOrder


class LambdaPreprocessCallback(Callback):
    """
    Preprocess your batch with specified function.

    Args:
            lambda_fn (Callable): Function to apply.
            input_keys (Union[List[str], str], optional): Keys in batch dict to apply function.
                Defaults to ``"logits"``.
            output_keys (Union[List[str], str], optional): Keys for output.
                If None then will apply function inplace to ``keys_to_apply``.
                Defaults to ``None``.

    Raises:
        TypeError: When keys_to_apply is not str or list.

    Examples:
        .. code-block:: python

            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from catalyst import dl

            # sample data
            num_users, num_features, num_items = int(1e4), int(1e1), 10
            X = torch.rand(num_users, num_features)
            y = (torch.rand(num_users, num_items) > 0.5).to(torch.float32)

            # pytorch loaders
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=32, num_workers=1)
            loaders = {"train": loader, "valid": loader}

            # model, criterion, optimizer, scheduler
            model = torch.nn.Linear(num_features, num_items)
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

            # model training
            runner = SupervisedRunner()
            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                loaders=loaders,
                num_epochs=3,
                verbose=True,
                callbacks=[
                    dl.LambdaPreprocessCallback(
                        input_keys="logits", output_keys="scores", lambda_fn=torch.sigmoid
                    ),
                    dl.CriterionCallback(
                        input_key="logits", target_key="targets", metric_key="loss"
                    ),
            # uncomment for extra metrics:
            #       dl.AUCCallback(
            #           input_key="scores", target_key="targets"
            #       ),
            #       dl.HitrateCallback(
            #           input_key="scores", target_key="targets", topk_args=(1, 3, 5)
            #       ),
            #       dl.MRRCallback(
            #           input_key="scores", target_key="targets", topk_args=(1, 3, 5)
            #       ),
            #       dl.MAPCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
            #       dl.NDCGCallback(
            #           input_key="scores", target_key="targets", topk_args=(1, 3, 5)
            #       ),
                    dl.OptimizerCallback(metric_key="loss"),
                    dl.SchedulerCallback(),
                    dl.CheckpointCallback(
                        logdir="./logs", loader_key="valid", metric_key="map01", minimize=False
                    ),
                ]
            )

    """

    def __init__(
        self,
        lambda_fn: Callable,
        input_keys: Union[List[str], str] = "logits",
        output_keys: Union[List[str], str] = None,
    ):
        """Wraps input for your callback with specified function.

        Args:
            lambda_fn (Callable): Function to apply.
            input_keys (Union[List[str], str], optional): Keys in batch dict to apply function.
                Defaults to ``"logits"``.
            output_keys (Union[List[str], str], optional): Keys for output.
                If None then will apply function inplace to ``keys_to_apply``.
                Defaults to ``None``.

        Raises:
            TypeError: When keys_to_apply is not str or list.
        """
        super().__init__(order=CallbackOrder.Internal)
        if not isinstance(input_keys, (list, str)):
            raise TypeError("keys to apply should be str or list of str.")
        elif isinstance(input_keys, str):
            input_keys = [input_keys]
        if output_keys is not None:
            if not isinstance(output_keys, (list, str)):
                raise TypeError("output keys should be str or list of str.")
            if isinstance(output_keys, str):
                output_keys = [output_keys]
        else:
            output_keys = input_keys
        self.keys_to_apply = input_keys
        self.output_keys = output_keys
        self.lambda_fn = lambda_fn
        self.handler: Union[None, Callable] = None

    def on_batch_end(self, runner) -> None:
        """
        On batch end action.

        Args:
            runner: runner for the experiment.

        Raises:
            TypeError: If lambda_fn output has a wrong type.

        """
        fn_inp = [runner.batch[key] for key in self.keys_to_apply]
        fn_output = self.lambda_fn(*fn_inp)
        if self.handler is None:
            if isinstance(fn_output, dict):
                self.handler = self._handle_output_dict
            if isinstance(fn_output, tuple):
                if len(fn_output) != len(self.output_keys):
                    raise TypeError(
                        "Unexpected output. "
                        "Expect function to return tuple same length as output_keys, "
                        f"which is {len(self.output_keys)}, "
                        f"but got output length of {len(fn_output)}"
                        "Use output_keys argument to specify output keys."
                    )
                self.handler = self._handle_output_tuple
            else:
                if len(self.output_keys) > 1:
                    raise TypeError(
                        "Unexpected output. "
                        "Expect function to return tuple or dict, but got "
                        f"{type(fn_output)}. "
                        "Use output_keys argument to specify output key."
                    )
                self.handler = self._handle_output_value
        runner.batch = self.handler(
            batch=runner.batch, function_output=fn_output, output_keys=self.output_keys
        )

    @staticmethod
    def _handle_output_tuple(
        batch: Dict[str, Any], function_output: Tuple[Any], output_keys: List[str]
    ) -> Dict[str, Any]:
        for out_idx, output_key in enumerate(output_keys):
            batch[output_key] = function_output[out_idx]
        return batch

    @staticmethod
    def _handle_output_dict(
        batch: Dict[str, Any], function_output: Dict[str, Any], output_keys: List[str]
    ) -> Dict[str, Any]:
        for output_key, output_value in function_output.items():
            batch[output_key] = output_value
        return batch

    @staticmethod
    def _handle_output_value(
        batch: Dict[str, Any], function_output: Any, output_keys: List[str],
    ):
        batch[output_keys[0]] = function_output
        return batch


__all__ = ["LambdaPreprocessCallback"]
