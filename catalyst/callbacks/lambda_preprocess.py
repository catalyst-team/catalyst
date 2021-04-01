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
                        keys_to_apply="logits", output_keys="scores", lambda_fn=torch.sigmoid
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
        keys_to_apply: Union[List[str], str] = "logits",
        output_keys: Union[List[str], str] = None,
    ):
        """Wraps input for your callback with specified function.

        Args:
            lambda_fn (Callable): Function to apply.
            keys_to_apply (Union[List[str], str], optional): Keys in batch dict to apply function.
                Defaults to ["s_hidden_states", "t_hidden_states"].
            output_keys (Union[List[str], str], optional): Keys for output.
                If None then will apply function inplace to ``keys_to_apply``.
                Defaults to None.

        Raises:
            TypeError: When keys_to_apply is not str or list.
        """
        super().__init__(order=CallbackOrder.Internal)
        if not isinstance(keys_to_apply, (list, str)):
            raise TypeError("keys to apply should be str or list of str.")
        if output_keys is not None:
            if not isinstance(output_keys, (list, str)):
                raise TypeError("output keys should be str or list of str.")
        self.keys_to_apply = keys_to_apply
        self.output_keys = output_keys
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
                if self.output_keys is not None:
                    if not isinstance(self.output_keys, list):
                        raise TypeError(
                            "Unexpected output from function. "
                            "For output key type string expected one element, got tuple."
                        )
                    iter_keys = self.output_keys
                else:
                    iter_keys = self.keys_to_apply
                for idx, key in enumerate(iter_keys):
                    batch[key] = fn_output[idx]
            elif isinstance(fn_output, dict):
                for outp_k, outp_v in fn_output.items():
                    batch[outp_k] = outp_v
            else:
                if self.output_keys is not None:
                    if not isinstance(self.output_keys, str):
                        raise TypeError(
                            "Unexpected output from function. "
                            "For output key type List[str] expected tuple, got one element."
                        )
                    output_key = self.output_keys
                else:
                    output_key = self.keys_to_apply
                batch[output_key] = fn_output
        elif isinstance(self.keys_to_apply, str):
            batch[self.keys_to_apply] = self.lambda_fn(self.keys_to_apply)
        runner.batch = batch


__all__ = ["LambdaPreprocessCallback"]
