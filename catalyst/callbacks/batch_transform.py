from typing import Any, Callable, Dict, List, Tuple, Union
from functools import partial

from catalyst.core import Callback, CallbackOrder, IRunner


class BatchTransformCallback(Callback):
    """
    Preprocess your batch with specified function.

    Args:
        lambda_fn (Callable): Function to apply.
        scope (str): ``"on_batch_end"`` or ``"on_batch_start"``
        input_key (Union[List[str], str, int], optional): Keys in batch dict to apply function.
            Defaults to ``None``.
        output_key (Union[List[str], str, int], optional): Keys for output.
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
        scope: str,
        input_key: Union[List[str], str, int] = None,
        output_key: Union[List[str], str, int] = None,
    ):
        """
        Preprocess your batch with specified function.

        Args:
            lambda_fn (Callable): Function to apply.
            scope (str): ``"on_batch_end"`` or ``"on_batch_start"``
            input_key (Union[List[str], str], optional): Keys in batch dict to apply function.
            output_key (Union[List[str], str], optional): Keys for output.
                If None then will apply function inplace to ``keys_to_apply``.
                Defaults to ``None``.

        Raises:
            TypeError: When keys_to_apply is not str or list.
        """
        super().__init__(order=CallbackOrder.Internal)
        if input_key is not None:
            if not isinstance(input_key, (list, str, int)):
                raise TypeError("input key should be str or list of str.")
            elif isinstance(input_key, (str, int)):
                input_key = [input_key]
            self.input_handler = self._handle_input_tuple
        else:
            self.input_handler = self._handle_input_dict

        output_key = output_key or input_key
        if output_key is not None:
            if not isinstance(output_key, (list, str, int)):
                raise TypeError("output key should be str or list of str.")
            if isinstance(output_key, (str, int)):
                self.output_handler = self._handle_output_value
                output_key = [output_key]
            else:
                self.output_handler = self._handle_output_tuple
        else:
            self.output_handler = self._handle_output_dict

        if isinstance(scope, str) and scope in ["on_batch_end", "on_batch_start"]:
            self.scope = scope
        else:
            raise TypeError('Expected scope to be on of the ["on_batch_end", "on_batch_start"]')
        self.input_key = input_key
        self.output_key = output_key
        self.lambda_fn = lambda_fn

    @staticmethod
    def _handle_input_tuple(batch, input_key):
        return [batch[key] for key in input_key]

    @staticmethod
    def _handle_input_dict(batch, _input_key):
        return batch

    @staticmethod
    def _handle_output_tuple(
        batch: Dict[str, Any], function_output: Tuple[Any], output_keys: List[str]
    ) -> Dict[str, Any]:
        for out_idx, output_key in enumerate(output_keys):
            batch[output_key] = function_output[out_idx]
        return batch

    @staticmethod
    def _handle_output_dict(
        batch: Dict[str, Any], function_output: Dict[str, Any], _output_keys: List[str]
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

    def _handle_batch(self, runner):
        fn_input = self.input_handler(runner.batch, self.input_key)
        fn_output = self.lambda_fn(*fn_input)

        runner.batch = self.output_handler(
            batch=runner.batch, function_output=fn_output, output_keys=self.output_key
        )

    def on_batch_start(self, runner: "IRunner") -> None:
        """
        On batch start action.

        Args:
            runner: runner for the experiment.
        """
        if self.scope == "on_batch_start":
            self._handle_batch(runner)

    def on_batch_end(self, runner) -> None:
        """
        On batch end action.

        Args:
            runner: runner for the experiment.
        """
        if self.scope == "on_batch_end":
            self._handle_batch(runner)


__all__ = ["BatchTransformCallback"]
