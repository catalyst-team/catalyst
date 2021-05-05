from typing import List

from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.metrics._cmc_score import CMCMetric, ReidCMCMetric


class CMCScoreCallback(LoaderMetricCallback):
    """
    Cumulative Matching Characteristics callback.

    This callback was designed to count
    cumulative matching characteristics.
    If current object is from query your dataset
    should output `True` in `is_query_key`
    and false if current object is from gallery.
    You can see `QueryGalleryDataset` in
    `catalyst.contrib.datasets.metric_learning` for more information.
    On batch end callback accumulate all embeddings

    Args:
        embeddings_key: embeddings key in output dict
        labels_key: labels key in output dict
        is_query_key: bool key True if current object is from query
        topk_args: specifies which cmc@K to log.
            [1] - cmc@1
            [1, 3] - cmc@1 and cmc@3
            [1, 3, 5] - cmc@1, cmc@3 and cmc@5
        prefix: metric prefix
        suffix: metric suffix

    .. note::

        You should use it with `ControlFlowCallback`
        and add all query/gallery sets to loaders.
        Loaders should contain "is_query" and "label" key.

    Examples:

    .. code-block:: python

        import os
        from torch.optim import Adam
        from torch.utils.data import DataLoader
        from catalyst import data, dl
        from catalyst.contrib import datasets, models, nn
        from catalyst.data.transforms import Compose, Normalize, ToTensor


        # 1. train and valid loaders
        transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MnistMLDataset(
            root=os.getcwd(), download=True, transform=transforms
            )
        sampler = data.BalanceBatchSampler(labels=train_dataset.get_labels(), p=5, k=10)
        train_loader = DataLoader(
            dataset=train_dataset, sampler=sampler, batch_size=sampler.batch_size
            )

        valid_dataset = datasets.MnistQGDataset(
            root=os.getcwd(), transform=transforms, gallery_fraq=0.2
        )
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1024)

        # 2. model and optimizer
        model = models.MnistSimpleNet(out_features=16)
        optimizer = Adam(model.parameters(), lr=0.001)

        # 3. criterion with triplets sampling
        sampler_inbatch = data.HardTripletsSampler(norm_required=False)
        criterion = nn.TripletMarginLossWithSampler(margin=0.5, sampler_inbatch=sampler_inbatch)

        # 4. training with catalyst Runner
        class CustomRunner(dl.SupervisedRunner):
            def handle_batch(self, batch) -> None:
                if self.is_train_loader:
                    images, targets = batch["features"].float(), batch["targets"].long()
                    features = self.model(images)
                    self.batch = {"embeddings": features, "targets": targets,}
                else:
                    images, targets, is_query = \
                        batch["features"].float(), \
                        batch["targets"].long(), \
                        batch["is_query"].bool()
                    features = self.model(images)
                    self.batch = {
                        "embeddings": features, "targets": targets, "is_query": is_query
                    }

        callbacks = [
            dl.ControlFlowCallback(
                dl.CriterionCallback(
                    input_key="embeddings", target_key="targets", metric_key="loss"
                ),
                loaders="train",
            ),
            dl.ControlFlowCallback(
                dl.CMCScoreCallback(
                    embeddings_key="embeddings",
                    labels_key="targets",
                    is_query_key="is_query",
                    topk_args=[1],
                ),
                loaders="valid",
            ),
            dl.PeriodicLoaderCallback(
                valid_loader_key="valid", valid_metric_key="cmc01", minimize=False, valid=2
            ),
        ]

        runner = CustomRunner(input_key="features", output_key="embeddings")
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            loaders={"train": train_loader, "valid": valid_loader},
            verbose=False,
            logdir="./logs",
            valid_loader="valid",
            valid_metric="cmc01",
            minimize_valid_metric=False,
            num_epochs=10,
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        embeddings_key: str,
        labels_key: str,
        is_query_key: str,
        topk_args: List[int] = None,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=CMCMetric(
                embeddings_key=embeddings_key,
                labels_key=labels_key,
                is_query_key=is_query_key,
                topk_args=topk_args,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=[embeddings_key, is_query_key],
            target_key=[labels_key],
        )


class ReidCMCScoreCallback(LoaderMetricCallback):
    """
    Cumulative Matching Characteristics callback for reID case.
    More information about cmc-based callbacks in CMCScoreCallback's docs.

    Args:
        embeddings_key: embeddings key in output dict
        pids_key: pids key in output dict
        cids_key: cids key in output dict
        is_query_key: bool key True if current object is from query
        topk_args: specifies which cmc@K to log.
            [1] - cmc@1
            [1, 3] - cmc@1 and cmc@3
            [1, 3, 5] - cmc@1, cmc@3 and cmc@5
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        embeddings_key: str,
        pids_key: str,
        cids_key: str,
        is_query_key: str,
        topk_args: List[int] = None,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=ReidCMCMetric(
                embeddings_key=embeddings_key,
                pids_key=pids_key,
                cids_key=cids_key,
                is_query_key=is_query_key,
                topk_args=topk_args,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=[embeddings_key, is_query_key],
            target_key=[pids_key, cids_key],
        )


__all__ = ["CMCScoreCallback", "ReidCMCScoreCallback"]
