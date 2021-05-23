from typing import Any, Dict, List

import torch

from catalyst.metrics._additive import AdditiveValueMetric
from catalyst.metrics._metric import ICallbackBatchMetric
from catalyst.metrics.functional._mrr import mrr


class MRRMetric(ICallbackBatchMetric):
    """
    Calculates the Mean Reciprocal Rank (MRR)
    score given model outputs and targets
    The precision metric summarizes the fraction of relevant items
    Computes mean value of map and it's approximate std value

    Args:
        topk_args: list of `topk` for mrr@topk computing
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import torch
        from catalyst import metrics

        outputs = torch.Tensor([
            [4.0, 2.0, 3.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
        ])
        targets = torch.tensor([
            [0, 0, 1.0, 1.0],
            [0, 0, 1.0, 1.0],
        ])
        metric = metrics.MRRMetric(topk_args=[1, 3])
        metric.reset()

        metric.update(outputs, targets)
        metric.compute()
        # ((0.5, 0.75), (0.0, 0.0))  # mean, std for @01, @03

        metric.compute_key_value()
        # {
        #     'mrr01': 0.5,
        #     'mrr03': 0.75,
        #     'mrr': 0.5,
        #     'mrr01/std': 0.0,
        #     'mrr03/std': 0.0,
        #     'mrr/std': 0.0
        # }

        metric.reset()
        metric(outputs, targets)
        # ((0.5, 0.75), (0.0, 0.0))  # mean, std for @01, @03

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
        runner = dl.SupervisedRunner(
            input_key="features", output_key="logits", target_key="targets", loss_key="loss"
        )
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            num_epochs=3,
            verbose=True,
            callbacks=[
                dl.BatchTransformCallback(
                    transform=torch.sigmoid,
                    scope="on_batch_end",
                    input_key="logits",
                    output_key="scores"
                ),
                dl.CriterionCallback(
                    input_key="logits", target_key="targets", metric_key="loss"
                ),
                dl.AUCCallback(input_key="scores", target_key="targets"),
                dl.HitrateCallback(
                    input_key="scores", target_key="targets", topk_args=(1, 3, 5)
                ),
                dl.MRRCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.MAPCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.NDCGCallback(input_key="scores", target_key="targets", topk_args=(1, 3, 5)),
                dl.OptimizerCallback(metric_key="loss"),
                dl.SchedulerCallback(),
                dl.CheckpointCallback(
                    logdir="./logs", loader_key="valid", metric_key="loss", minimize=True
                ),
            ]
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        topk_args: List[int] = None,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init MRRMetric"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name_mean = f"{self.prefix}mrr{self.suffix}"
        self.metric_name_std = f"{self.prefix}mrr{self.suffix}/std"
        self.topk_args: List[int] = topk_args or [1]
        self.additive_metrics: List[AdditiveValueMetric] = [
            AdditiveValueMetric() for _ in range(len(self.topk_args))
        ]

    def reset(self) -> None:
        """Reset all fields"""
        for metric in self.additive_metrics:
            metric.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """
        Update metric value with map for new data and return intermediate metrics values.

        Args:
            logits (torch.Tensor): tensor of logits
            targets (torch.Tensor): tensor of targets

        Returns:
            list of map@k values
        """
        values = mrr(logits, targets, topk=self.topk_args)
        values = [v.item() for v in values]
        for value, metric in zip(values, self.additive_metrics):
            metric.update(value, len(targets))
        return values

    def update_key_value(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Update metric value with mrr for new data and return intermediate metrics
        values in key-value format.

        Args:
            logits (torch.Tensor): tensor of logits
            targets (torch.Tensor): tensor of targets

        Returns:
            dict of mrr@k values
        """
        values = self.update(logits=logits, targets=targets)
        output = {
            f"{self.prefix}mrr{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, values)
        }
        output[self.metric_name_mean] = output[f"{self.prefix}mrr01{self.suffix}"]
        return output

    def compute(self) -> Any:
        """
        Compute mrr for all data

        Returns:
            list of mean values, list of std values
        """
        means, stds = zip(*(metric.compute() for metric in self.additive_metrics))
        return means, stds

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute mrr for all data and return results in key-value format

        Returns:
            dict of metrics
        """
        means, stds = self.compute()
        output_mean = {
            f"{self.prefix}mrr{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, means)
        }
        output_std = {
            f"{self.prefix}mrr{key:02d}{self.suffix}/std": value
            for key, value in zip(self.topk_args, stds)
        }
        output_mean[self.metric_name_mean] = output_mean[f"{self.prefix}mrr01{self.suffix}"]
        output_std[self.metric_name_std] = output_std[f"{self.prefix}mrr01{self.suffix}/std"]
        return {**output_mean, **output_std}


__all__ = ["MRRMetric"]
