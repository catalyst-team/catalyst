from typing import Callable, List, Union

from catalyst.core import Callback, CallbackOrder, IRunner


class BatchTransformCallback(Callback):
    """
    Preprocess your batch with specified function.

    Args:
        transform (Callable): Function to apply.
        scope (str): ``"on_batch_end"`` (post-processing model output) or
            ``"on_batch_start"`` (pre-processing model input).
        input_key (Union[List[str], str, int], optional): Keys in batch dict to apply function.
            Defaults to ``None``.
        output_key (Union[List[str], str, int], optional): Keys for output.
            If None then will apply function inplace to ``keys_to_apply``.
            Defaults to ``None``.

    Raises:
        TypeError: When keys_to_apply is not str or a list.

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
                        input_key="logits", output_key="scores", transform=torch.sigmoid
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
        transform: Callable,
        scope: str,
        input_key: Union[List[str], str] = None,
        output_key: Union[List[str], str] = None,
    ):
        """
        Preprocess your batch with specified function.

        Args:
            transform (Callable): Function to apply.
            scope (str): ``"on_batch_end"`` (post-processing model output) or
                ``"on_batch_start"`` (pre-processing model input).
            input_key (Union[List[str], str], optional): Keys in batch dict to apply function.
            output_key (Union[List[str], str], optional): Keys for output.
                If None then will apply function inplace to ``keys_to_apply``.
                Defaults to ``None``.

        Raises:
            TypeError: When keys_to_apply is not str or a list.
        """
        super().__init__(order=CallbackOrder.Internal)
        if input_key is not None:
            if not isinstance(input_key, (list, str)):
                raise TypeError("input key should be str or a list of str.")
            elif isinstance(input_key, str):
                input_key = [input_key]
            self._handle_batch = self._handle_tuples
        else:
            self._handle_batch = self._handle_dicts

        output_key = output_key or input_key
        if output_key is not None:
            if input_key is None:
                raise TypeError("You should define input_key in " "case if output_key is not None")
            if not isinstance(output_key, (list, str)):
                raise TypeError("output key should be str or a list of str.")
            if isinstance(output_key, str):
                output_key = [output_key]

        if isinstance(scope, str) and scope in ["on_batch_end", "on_batch_start"]:
            self.scope = scope
        else:
            raise TypeError('Expected scope to be on of the ["on_batch_end", "on_batch_start"]')
        self.input_key = input_key
        self.output_key = output_key
        self.transform = transform

    def _handle_dicts(self, runner):
        runner.batch = self.transform(runner.batch)

    def _handle_tuples(self, runner):
        batch_in = [runner.batch[key] for key in self.input_key]
        batch_out = self.handler(batch_in)
        runner.batch.update(**{key: value for key, value in zip(self.output_key, batch_out)})

    def on_batch_start(self, runner: "IRunner") -> None:
        """
        On batch start action.

        Args:
            runner: runner for the experiment.
        """
        if self.scope == "on_batch_start":
            self._handle_batch(runner)

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action.

        Args:
            runner: runner for the experiment.
        """
        if self.scope == "on_batch_end":
            self._handle_batch(runner)


__all__ = ["BatchTransformCallback"]
