from typing import Dict, Tuple

import torch

from catalyst.metrics._metric import ICallbackLoaderMetric
from catalyst.metrics.functional._auc import auc, binary_auc
from catalyst.metrics.functional._misc import process_multilabel_components
from catalyst.utils.distributed import all_gather, get_rank


class AUCMetric(ICallbackLoaderMetric):
    """AUC metric,

    Args:
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix

    .. warning::

        This metric is under API improvement.

    Examples:

    .. code-block:: python

        import torch
        from catalyst import metrics

        scores = torch.tensor([
            [0.9, 0.1],
            [0.1, 0.9],
        ])
        targets = torch.tensor([
            [1, 0],
            [0, 1],
        ])
        metric = metrics.AUCMetric()

        # for efficient statistics storage
        metric.reset(num_batches=1, num_samples=len(scores))
        metric.update(scores, targets)
        metric.compute()
        # (
        #     tensor([1., 1.])  # per class
        #     1.0,              # micro
        #     1.0,              # macro
        #     1.0               # weighted
        # )

        metric.compute_key_value()
        # {
        #     'auc': 1.0,
        #     'auc/_micro': 1.0,
        #     'auc/_macro': 1.0,
        #     'auc/_weighted': 1.0
        #     'auc/class_00': 1.0,
        #     'auc/class_01': 1.0,
        # }

        metric.reset(num_batches=1, num_samples=len(scores))
        metric(scores, targets)
        # (
        #     tensor([1., 1.])  # per class
        #     1.0,              # micro
        #     1.0,              # macro
        #     1.0               # weighted
        # )

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

        # sample data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples,) * num_classes).to(torch.int64)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = torch.nn.CrossEntropyLoss()
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
            logdir="./logdir",
            num_epochs=3,
            valid_loader="valid",
            valid_metric="accuracy03",
            minimize_valid_metric=False,
            verbose=True,
            callbacks=[
                dl.AccuracyCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.PrecisionRecallF1SupportCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.AUCCallback(input_key="logits", target_key="targets"),
            ],
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        """Init."""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name = f"{self.prefix}auc{self.suffix}"
        self.scores = []
        self.targets = []
        self._is_ddp = get_rank() > -1

    def reset(self, num_batches, num_samples) -> None:
        """Resets all fields"""
        self._is_ddp = get_rank() > -1
        self.scores = []
        self.targets = []

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates metric value with statistics for new data.

        Args:
            scores: tensor with scores
            targets: tensor with targets
        """
        self.scores.append(scores.cpu().detach())
        self.targets.append(targets.cpu().detach())

    def compute(self) -> Tuple[torch.Tensor, float, float, float]:
        """Computes the AUC metric based on saved statistics."""
        targets = torch.cat(self.targets)
        scores = torch.cat(self.scores)

        # @TODO: ddp hotfix, could be done better
        if self._is_ddp:
            scores = torch.cat(all_gather(scores))
            targets = torch.cat(all_gather(targets))

        scores, targets, _ = process_multilabel_components(outputs=scores, targets=targets)
        per_class = auc(scores=scores, targets=targets)
        micro = binary_auc(scores=scores.view(-1), targets=targets.view(-1))[0]
        macro = per_class.mean().item()
        weights = targets.sum(axis=0) / len(targets)
        weighted = (per_class * weights).sum().item()
        return per_class, micro, macro, weighted

    def compute_key_value(self) -> Dict[str, float]:
        """Computes the AUC metric based on saved statistics and returns key-value results."""
        per_class_auc, micro_auc, macro_auc, weighted_auc = self.compute()
        output = {
            f"{self.metric_name}/class_{i:02d}": value.item()
            for i, value in enumerate(per_class_auc)
        }
        output[f"{self.metric_name}/_micro"] = micro_auc
        output[self.metric_name] = macro_auc
        output[f"{self.metric_name}/_macro"] = macro_auc
        output[f"{self.metric_name}/_weighted"] = weighted_auc
        return output


__all__ = ["AUCMetric"]
