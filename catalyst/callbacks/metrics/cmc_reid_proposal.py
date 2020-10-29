from typing import List

from catalyst.core import Callback
from catalyst.metrics.functional import get_default_topk_args
from catalyst.metrics.cmc_score import cmc_score, masked_cmc_score


class AccamulatorCallback(Callback):

    def __init__(self,
                 input_keys: List[str],
                 output_keys: List[str],
                 ):
        # умеет копить примиты (их кладем в одномерный массив)
        # и одномерные массивы (кладем в таблицу)
        super().__init__(order=CallbackOrder.Metric)

        assert not set(input_keys).intersection(set(output_keys))

        self._input_keys = input_keys
        self._output_keys = output_keys

        self._cur_ids = None  # todo
        self._storage = None

    def _check_completness(self):
        pass  # todo

    def on_loader_start(self, runner: "IRunner") -> None:
        for k, source in [(self._input_keys, runner.input), (self._output_keys, self.output)]:
            _, dim = source[k].shape
            self._storage[k] = torch.empty(size=(storage_size, dim))

    def on_loader_end(self, runner: "IRunner") -> None:
        self._check_completness()

    def on_batch_end(self, runner: "IRunner") -> None:
        # self.cur_ids + range(len(runner.output))
        for k, source in [(self._input_keys, runner.input), (self._output_keys, self.output)]:
            self._storage[k][cur_ids] = source[k]


class CMCScoreCallback(AccamulatorCallback):

    def __init__(
        self,
        embeddings_key: str = "logits",
        labels_key: str = "targets",
        is_query_key: str = "is_query",
        prefix: str = "cmc",
        topk_args: List[int] = None,
        num_classes: int = None,
    ):
        super().__init__(input_keys=[labels_key, is_query_key],
                         output_keys=[embeddings_key],
                         )

        self._embedding_key = embeddings_key
        self._labels_key = labels_key
        self._is_query_key = is_query_key

        self._prefix = prefix
        self.list_args = topk_args or get_default_topk_args(num_classes)

        self._metric_fn = cmc_score

    def on_loader_end(self, runner: "IRunner") -> None:
        self._check_completness()

        query_mask = self.storage[self._is_query_key]
        gallery_mask = ~query_mask

        gallery_labels = self._storage[self._labels_key][gallery_mask]
        query_labels = self._storage[self._labels_key][query_mask]
        query_embeddings = self._storage[self._embedding_key][query_mask]
        gallery_embeddings = self._storage[self._embedding_key][gallery_mask]

        conformity_matrix = gallery_labels == query_labels.reshape(-1, 1)

        for key in self.list_args:
            metric = self._metric_fn(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                conformity_matrix=conformity_matrix,
                topk=key,
            )
            runner.loader_metrics[f"{self._prefix}{key:02}"] = metric


class ReidCMCScoreCallback(AccamulatorCallback):

    def __init__(
            self,
            embeddings_key: str = "logits",
            pids_key: str = "pid",
            cids_key: str = "cid",
            is_query_key: str = "is_query",
            prefix: str = "cmc",
            topk_args: List[int] = None,
            num_classes: int = None,
    ):

        super().__init__(input_keys=[pids_key, cids_key, is_query_key],
                         output_keys=[embeddings_key],
                         )

        self._embedding_key = embeddings_key
        self._pids_key = pids_key
        self._cids_key = cids_key
        self._is_query_key = is_query_key

        self._prefix = prefix
        self.list_args = topk_args or get_default_topk_args(num_classes)

        self._metric_fn = masked_cmc_score

    def on_loader_end(self, runner: "IRunner") -> None:
        self._check_completness()

        query_mask = self.storage[self._is_query_key]
        gallery_mask = ~query_mask

        gallery_pids = self._storage[self._pids_key][gallery_mask]
        query_pids = self._storage[self._pids_key][query_mask]
        gallery_cids = self._storage[self._cids_key][gallery_mask]
        query_cids = self._storage[self._cids_key][query_mask]

        pid_conformity_matrix = gallery_pids == query_pids.reshape(-1, 1)
        cid_conformity_matrix = gallery_cids == query_cids.reshape(-1, 1)
        available_samples = ~(pid_conformity_matrix * cid_conformity_matrix)

        if (available_samples.max(dim=1).values == 0).any():
            ValueError(
                "There is a sample in query that has no relevant samples "
                "in gallery."
            )

        for key in self.list_args:
            metric = self._metric_fn(
                query_embeddings=self._storage[self._embedding_key][query_mask],
                gallery_embeddings=self._storage[self._embedding_key][gallery_mask],
                conformity_matrix=pid_conformity_matrix,
                available_samples=available_samples,
                topk=key,
            )
            runner.loader_metrics[f"{self._prefix}{key:02}"] = metric
