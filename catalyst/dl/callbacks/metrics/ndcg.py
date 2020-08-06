from typing import List

from catalyst.core import MultiMetricCallback
from catalyst.utils import metrics


class NdcgCallback(MultiMetricCallback):
    """NDCG metric callback.
    Computes ndcg@topk for the specified values of `topk`.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "ndcg",
        ndcg_args: List[int] = None,
    ):
        """
        Args:
            input_key (str): input key to use for ndcg calculation;
                specifies our `y_true`
            output_key (str): output key to use for ndcg calculation;
                specifies our `y_pred`
            prefix (str): key for the metric's name
            ndcg_args (List[int]): specifies which ndcg@K to log:
                [1] - ndcg at 1
                [1, 3] - ndcg at 1 and 3
                [1, 3, 5] - ndcg at 1, 3 and 5
        """
        list_args = ndcg_args or (10,)

        super().__init__(
            prefix=prefix,
            metric_fn=metrics.ndcg,
            input_key=input_key,
            output_key=output_key,
            topk=list_args,
        )


__all__ = ["NdcgCallback"]
