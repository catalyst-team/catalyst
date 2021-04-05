from typing import Dict, Iterable, List, Optional

import torch

from catalyst.metrics._metric import AccumulationMetric
from catalyst.metrics.functional._cmc_score import cmc_score, masked_cmc_score
from catalyst.utils.distributed import get_rank


class CMCMetric(AccumulationMetric):
    """Cumulative Matching Characteristics

    Args:
        embeddings_key: key of embedding tensor in batch
        labels_key: key of label tensor in batch
        is_query_key: key of query flag tensor in batch
        topk_args: list of k, specifies which cmc@k should be calculated
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix

    Examples:
        >>> from collections import OrderedDict
        >>> from torch.optim import Adam
        >>> from torch.utils.data import DataLoader
        >>> from catalyst.contrib import nn
        >>> from catalyst.contrib.datasets import MnistMLDataset, MnistQGDataset
        >>> from catalyst.data import BalanceBatchSampler, HardTripletsSampler
        >>> from catalyst.dl import ControlFlowCallback, LoaderMetricCallback, SupervisedRunner
        >>> from catalyst.metrics import CMCMetric
        >>>
        >>> dataset_root = "."
        >>>
        >>> # download dataset for train and val, create loaders
        >>> dataset_train = MnistMLDataset(root=dataset_root, download=True, transform=None)
        >>> sampler = BalanceBatchSampler(labels=dataset_train.get_labels(), p=5, k=10)
        >>> train_loader = DataLoader(
        >>>     dataset=dataset_train, sampler=sampler, batch_size=sampler.batch_size
        >>> )
        >>> dataset_valid = MnistQGDataset(root=dataset_root, transform=None, gallery_fraq=0.2)
        >>> valid_loader = DataLoader(dataset=dataset_valid, batch_size=1024)
        >>>
        >>> # model, optimizer, criterion
        >>> model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 100))
        >>> optimizer = Adam(model.parameters())
        >>> sampler_inbatch = HardTripletsSampler(norm_required=False)
        >>> criterion = nn.TripletMarginLossWithSampler(
        >>>     margin=0.5, sampler_inbatch=sampler_inbatch
        >>> )
        >>>
        >>> # batch data processing
        >>> class CustomRunner(SupervisedRunner):
        >>>     def handle_batch(self, batch):
        >>>         if self.is_train_loader:
        >>>             images, targets = batch["features"].float(), batch["targets"].long()
        >>>             features = model(images)
        >>>             self.batch = {
        >>>                 "embeddings": features,
        >>>                 "targets": targets,
        >>>             }
        >>>         else:
        >>>             images, targets, is_query = (
        >>>                 batch["features"].float(),
        >>>                 batch["targets"].long(),
        >>>                 batch["is_query"].bool(),
        >>>             )
        >>>             features = model(images)
        >>>             self.batch = {
        >>>                 "embeddings": features,
        >>>                 "targets": targets,
        >>>                 "is_query": is_query,
        >>>             }
        >>>
        >>> # training
        >>> runner = CustomRunner(input_key="features", output_key="embeddings")
        >>> runner.train(
        >>>     model=model,
        >>>     criterion=criterion,
        >>>     optimizer=optimizer,
        >>>     callbacks=OrderedDict(
        >>>         {
        >>>             "cmc": ControlFlowCallback(
        >>>                 LoaderMetricCallback(
        >>>                     CMCMetric(
        >>>                         embeddings_key="embeddings",
        >>>                         labels_key="targets",
        >>>                         is_query_key="is_query",
        >>>                         topk_args=(1, 3)
        >>>                     ),
        >>>                     input_key=["embeddings", "is_query"],
        >>>                     target_key=["targets"]
        >>>                 ),
        >>>                 loaders="valid",
        >>>             ),
        >>>         }
        >>>     ),
        >>>     loaders=OrderedDict({"train": train_loader, "valid": valid_loader}),
        >>>     valid_loader="valid",
        >>>     valid_metric="cmc01",
        >>>     minimize_valid_metric=False,
        >>>     logdir="./logs",
        >>>     verbose=True,
        >>>     num_epochs=3,
        >>> )
    """

    def __init__(
        self,
        embeddings_key: str,
        labels_key: str,
        is_query_key: str,
        topk_args: Iterable[int] = None,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """Init CMCMetric"""
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            accumulative_fields=[embeddings_key, labels_key, is_query_key],
        )
        self.embeddings_key = embeddings_key
        self.labels_key = labels_key
        self.is_query_key = is_query_key
        self.topk_args = topk_args or (1,)
        self.metric_name = f"{self.prefix}cmc{self.suffix}"

    def reset(self, num_batches: int, num_samples: int) -> None:
        """
        Reset metrics fields

        Args:
            num_batches: expected number of batches
            num_samples: expected number of samples to accumulate
        """
        super().reset(num_batches, num_samples)
        assert get_rank() < 0, "No DDP support implemented yet"

    def compute(self) -> List[float]:
        """
        Compute cmc@k metrics with all the accumulated data for all k.

        Returns:
            list of metrics values
        """
        query_mask = (self.storage[self.is_query_key] == 1).to(torch.bool)

        embeddings = self.storage[self.embeddings_key].float()
        labels = self.storage[self.labels_key]

        query_embeddings = embeddings[query_mask]
        query_labels = labels[query_mask]

        gallery_embeddings = embeddings[~query_mask]
        gallery_labels = labels[~query_mask]

        conformity_matrix = (gallery_labels == query_labels.reshape(-1, 1)).to(torch.bool)

        metrics = []
        for k in self.topk_args:
            value = cmc_score(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                conformity_matrix=conformity_matrix,
                topk=k,
            )
            metrics.append(value)

        return metrics

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute cmc@k metrics with all the accumulated data for all k.

        Returns:
            metrics values in key-value format
        """
        values = self.compute()
        kv_metrics = {
            f"{self.metric_name}{k:02d}": value for k, value in zip(self.topk_args, values)
        }
        return kv_metrics


class ReidCMCMetric(AccumulationMetric):
    """Cumulative Matching Characteristics for Reid case

    Args:
        embeddings_key: key of embedding tensor in batch
        pids_key: key of pids tensor in batch
        cids_key: key of cids tensor in batch
        is_query_key: key of query flag tensor in batch
        topk_args: list of k, specifies which cmc@k should be calculated
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        embeddings_key: str,
        pids_key: str,
        cids_key: str,
        is_query_key: str,
        topk_args: Iterable[int] = None,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """Init CMCMetric"""
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            accumulative_fields=[embeddings_key, pids_key, cids_key, is_query_key],
        )
        self.embeddings_key = embeddings_key
        self.pids_key = pids_key
        self.cids_key = cids_key
        self.is_query_key = is_query_key
        self.topk_args = topk_args or (1,)
        self.metric_name = f"{self.prefix}cmc{self.suffix}"

    def reset(self, num_batches: int, num_samples: int) -> None:
        """
        Reset metrics fields

        Args:
            num_batches: expected number of batches
            num_samples: expected number of samples to accumulate
        """
        super().reset(num_batches, num_samples)
        assert get_rank() < 0, "No DDP support implemented yet"

    def compute(self) -> List[float]:
        """
        Compute cmc@k metrics with all the accumulated data for all k.

        Returns:
            list of metrics values

        Raises:
            ValueError: if there are samples in query that have no relevant samples in gallery
        """
        query_mask = (self.storage[self.is_query_key] == 1).to(torch.bool)

        embeddings = self.storage[self.embeddings_key].float()
        pids = self.storage[self.pids_key]
        cids = self.storage[self.cids_key]

        query_embeddings = embeddings[query_mask]
        query_pids = pids[query_mask]
        query_cids = cids[query_mask]

        gallery_embeddings = embeddings[~query_mask]
        gallery_pids = pids[~query_mask]
        gallery_cids = cids[~query_mask]

        pid_conformity_matrix = (gallery_pids == query_pids.reshape(-1, 1)).bool()
        cid_conformity_matrix = (gallery_cids == query_cids.reshape(-1, 1)).bool()

        # Now we are going to generate a mask that should show if
        # a sample from gallery can be used during model scoring on the query
        # sample.
        # There is only one case when the label shouldn't be used for:
        # if query sample is a photo of the person pid_i taken from camera
        # cam_j and the gallery sample is a photo of the same person pid_i
        # from the same camera cam_j. All other cases are available.
        available_samples = ~(pid_conformity_matrix * cid_conformity_matrix).bool()

        if (available_samples.max(dim=1).values == 0).any():
            raise ValueError("There is a sample in query that has no relevant samples in gallery.")

        metrics = []
        for k in self.topk_args:
            value = masked_cmc_score(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                conformity_matrix=pid_conformity_matrix,
                available_samples=available_samples,
                topk=k,
            )
            metrics.append(value)

        return metrics

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute cmc@k metrics with all the accumulated data for all k.

        Returns:
            metrics values in key-value format
        """
        values = self.compute()
        kv_metrics = {
            f"{self.metric_name}{k:02d}": value for k, value in zip(self.topk_args, values)
        }
        return kv_metrics


__all__ = ["CMCMetric", "ReidCMCMetric"]
